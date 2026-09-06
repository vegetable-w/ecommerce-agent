import logging

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.runnables import Runnable

from app.config import settings
from app.core.llm import get_chat_model
from app.core.prompts import EXTRACT_PROMPT
from app.schemas.extract import AfterSalesTicket, ExtractRequest

logger = logging.getLogger(__name__)
router = APIRouter()


def get_extractor() -> Runnable:
    model = get_chat_model()
    # include_raw=Trueにより、パース失敗は例外にならずraw/parsed/parsing_errorの
    # 封筒(dict)として返る。これにより「上流障害」と「応答の解析失敗」を
    # extract()側で区別できる
    return EXTRACT_PROMPT | model.with_structured_output(
        AfterSalesTicket, method=settings.extract_method, include_raw=True
    )


@router.post("/api/extract", response_model=AfterSalesTicket)
async def extract(
    req: ExtractRequest, extractor: Runnable = Depends(get_extractor)
) -> AfterSalesTicket:
    try:
        result = await extractor.ainvoke({"text": req.text})
    except Exception as exc:
        # 上流への到達・認証・タイムアウトなど、輸送レベルの失敗はここに来る
        logger.exception("構造化抽出に失敗")
        raise HTTPException(status_code=502, detail="上流モデルは一時的に利用できません。しばらくしてから再試行してください") from exc

    if result["parsing_error"] is not None:
        # raw.contentはモデルの応答内容であり、ユーザーの入力(req.text)ではないため、
        # ここでログに残すことは「ユーザーデータをログに含めない」原則への
        # 意図的かつ限定的な例外である。診断に不可欠なので、この行を原則違反として
        # 削除しないこと
        logger.error("抽出結果の解析に失敗 raw=%r", result["raw"].content)
        # プロンプトやスキーマ、上流のレスポンス形状に起因するサーバー側の欠陥であり、
        # 再試行しても決定的に同じ結果になるため、502(上流一時利用不可)ではなく500とする
        raise HTTPException(status_code=500, detail="抽出結果の解析に失敗しました")

    return result["parsed"]
