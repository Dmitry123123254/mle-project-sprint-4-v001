# api_app.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI
from pydantic import BaseModel

from recommendation_service import Recommendations

# Загрузка переменных окружения из .env файла
load_dotenv()

class RecommendRequest(BaseModel):
    user_id: int
    k: int = 20
    recent_tracks: Optional[List[int]] = None  # онлайн-история: недавно прослушанные треки

class RecommendResponse(BaseModel):
    user_id: int
    tracks: List[int]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка рекомендаций из S3 при старте приложения"""
    rec = Recommendations()
    rec.load("final_ranked")
    rec.load("personal_als")
    rec.load("top_popular")
    app.state.rec = rec
    yield
    rec.stats()  # выводим статистику при завершении
    del app.state.rec

app = FastAPI(
    title="Music Recommendations API",
    description="Микросервис рекомендаций с учётом истории пользователя",
    lifespan=lifespan
)

@app.get("/health")
def health() -> Dict[str, Any]:
    """Проверка работоспособности сервиса"""
    return {"status": "ok"}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    """
    Возвращает рекомендации для пользователя.
    
    - Если пользователь известен (есть история): персональные рекомендации
    - Если пользователь новый (нет истории): топ-популярные треки
    - Если переданы recent_tracks: учёт онлайн-истории для улучшения рекомендаций
    """
    rec: Recommendations = app.state.rec
    
    if req.recent_tracks:
        # Смешивание офлайн + онлайн
        tracks = rec.get_with_online(
            user_id=req.user_id,
            k=req.k,
            recent_tracks=req.recent_tracks,
            alpha=0.3  # 30% веса онлайн-сигналу
        )
    else:
        # Только офлайн-рекомендации
        tracks = rec.get_offline(user_id=req.user_id, k=req.k)
    
    return RecommendResponse(user_id=req.user_id, tracks=tracks)

@app.get("/stats")
def get_stats() -> Dict[str, Any]:
    """Статистика использования рекомендаций"""
    return app.state.rec._stats
