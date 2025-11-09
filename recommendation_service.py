# recommendation_service.py
import os
from dotenv import load_dotenv
import io
import logging as logger
from typing import Dict, List, Optional
import numpy as np

import boto3
from botocore.client import Config
import pandas as pd

S3_KEYS: Dict[str, str] = {
    "top_popular":  "recsys/recommendations/top_popular.parquet",      # ['track_id','rank','listen_count']
    "personal_als": "recsys/recommendations/personal_als.parquet",     # ['user_id','track_id','score','rank']
    "final_ranked": "recsys/recommendations/recommendations.parquet",  # ['user_id','track_id','score','rank']
}

def _require_env(*names: str) -> None:
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise ValueError(f"Не заданы переменные окружения: {', '.join(missing)}")

# Загрузка переменных окружения из .env файла
load_dotenv()

def make_s3_client():
    _require_env("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "student_s3_bucket")
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        endpoint_url="https://storage.yandexcloud.net",
        config=Config(signature_version="s3v4", s3={"payload_signing_enabled": False}),
    )

def read_parquet_s3(s3_client, bucket: str, key: str, **kwargs) -> pd.DataFrame:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    buf = io.BytesIO(obj["Body"].read())
    return pd.read_parquet(buf, **kwargs)  # pyarrow/fastparquet [web:200]

class Recommendations:
    def __init__(self):
        self._recs = {"final_ranked": None, "personal_als": None, "top_popular": None}
        self._stats = {
            "request_personal_count": 0,     # пользователи с историей
            "request_default_count": 0,      # пользователи без истории
            "request_with_online_count": 0,  # запросы с онлайн-сигналами
        }
        self._s3 = make_s3_client()
        self._bucket = os.getenv("student_s3_bucket")

    def load(self, rtype: str, **kwargs):
        """Загружает один из типов рекомендаций из S3"""
        if rtype not in S3_KEYS:
            raise ValueError(f"rtype must be in {list(S3_KEYS.keys())}")
        key = S3_KEYS[rtype]
        logger.info(f"Loading '{rtype}' from s3://{self._bucket}/{key}")
        df = read_parquet_s3(self._s3, self._bucket, key, **kwargs)
        if rtype in {"final_ranked", "personal_als"} and "user_id" in df.columns:
            df = df.set_index("user_id")
        self._recs[rtype] = df
        logger.info(f"Loaded '{rtype}' with {len(df)} rows")

    def _order_personal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Сортировка персональных рекомендаций: score↓ → rank↑"""
        if "score" in df.columns and "rank" in df.columns:
            return df.sort_values(["score", "rank"], ascending=[False, True])
        if "score" in df.columns:
            return df.sort_values("score", ascending=False)
        if "rank" in df.columns:
            return df.sort_values("rank", ascending=True)
        return df

    def _order_top(self, df: pd.DataFrame) -> pd.DataFrame:
        """Сортировка топ-популярных: listen_count↓ → rank↑"""
        if "listen_count" in df.columns and "rank" in df.columns:
            return df.sort_values(["listen_count", "rank"], ascending=[False, True])
        if "listen_count" in df.columns:
            return df.sort_values("listen_count", ascending=False)
        if "rank" in df.columns:
            return df.sort_values("rank", ascending=True)
        return df

    def get_offline(self, user_id: int, k: int = 100) -> List[int]:
        """
        Возвращает офлайн-рекомендации для пользователя.
        Приоритет: final_ranked → personal_als → top_popular (для новых пользователей).
        Это учитывает всю историю пользователя, собранную на этапе офлайн-обучения.
        """
        # 1) Финально ранжированные персональные (учитывают всю историю)
        fr = self._recs["final_ranked"]
        if fr is not None:
            try:
                recs = fr.loc[user_id]
                if isinstance(recs, pd.Series):
                    recs = recs.to_frame().T
                recs = self._order_personal(recs)
                tracks = recs["track_id"].head(k).tolist()
                self._stats["request_personal_count"] += 1
                logger.info(f"User {user_id}: returned {len(tracks)} tracks from final_ranked")
                return tracks
            except KeyError:
                pass

        # 2) Персональные ALS (учитывают историю для collaborative filtering)
        pa = self._recs["personal_als"]
        if pa is not None:
            try:
                recs = pa.loc[user_id]
                if isinstance(recs, pd.Series):
                    recs = recs.to_frame().T
                recs = self._order_personal(recs)
                tracks = recs["track_id"].head(k).tolist()
                self._stats["request_personal_count"] += 1
                logger.info(f"User {user_id}: returned {len(tracks)} tracks from personal_als")
                return tracks
            except KeyError:
                pass

        # 3) Топ-популярные (для пользователей без истории)
        tp = self._recs["top_popular"]
        if tp is not None:
            recs = self._order_top(tp)
            tracks = recs["track_id"].head(k).tolist()
            self._stats["request_default_count"] += 1
            logger.info(f"User {user_id}: no history, returned {len(tracks)} tracks from top_popular")
            return tracks

        logger.error("No recommendations available")
        return []

    def get_with_online(
        self, 
        user_id: int, 
        k: int = 100, 
        recent_tracks: Optional[List[int]] = None,
        alpha: float = 0.3
    ) -> List[int]:
        """
        Возвращает рекомендации с учетом онлайн-истории (недавно прослушанные треки).
        Смешивает офлайн-оценки с онлайн-сигналом (популярность недавних треков).
        alpha: вес онлайн-сигнала (0..1), по умолчанию 0.3.
        
        Логика:
        - Берём персональные офлайн-рекомендации
        - Если есть recent_tracks, вычисляем "онлайн-оценку" (насколько похожи кандидаты на недавние)
        - Смешиваем: score_final = (1-alpha)*score_offline + alpha*score_online
        - Сортируем и возвращаем top-K
        """
        # Получаем офлайн-кандидаты (расширенный список для последующей фильтрации)
        offline = self.get_offline(user_id, k=k*5)
        if not offline:
            return []

        # Если нет онлайн-истории или пользователь новый, возвращаем офлайн
        if not recent_tracks:
            return offline[:k]

        self._stats["request_with_online_count"] += 1

        # Формируем DataFrame кандидатов
        cands_df = pd.DataFrame({"track_id": offline})
        
        # Простой онлайн-сигнал: бустим треки, похожие на недавние
        # (в реальности здесь можно использовать embeddings, жанры, исполнителей и т.п.)
        cands_df["online_boost"] = cands_df["track_id"].apply(
            lambda tid: 1.0 if tid in recent_tracks else 0.0
        )
        
        # Нормализуем: офлайн-позиция (чем раньше в списке, тем выше оценка)
        cands_df["offline_score"] = 1.0 / (cands_df.index + 1)
        
        # Смешиваем оценки
        cands_df["final_score"] = (
            (1 - alpha) * cands_df["offline_score"] + 
            alpha * cands_df["online_boost"]
        )
        
        # Сортируем и берём top-K
        cands_df = cands_df.sort_values("final_score", ascending=False)
        result = cands_df["track_id"].head(k).tolist()
        
        logger.info(
            f"User {user_id}: blended offline+online, "
            f"recent_tracks={len(recent_tracks)}, returned {len(result)} tracks"
        )
        return result

    def stats(self):
        """Выводит статистику по типам запросов"""
        logger.info("=== Recommendation Stats ===")
        for name, value in self._stats.items():
            logger.info(f"{name:<30} {value}")
