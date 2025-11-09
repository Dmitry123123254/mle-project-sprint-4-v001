# test_service.py
import os
from dotenv import load_dotenv
import io
import json
from fastapi.testclient import TestClient
import pandas as pd
import boto3
from botocore.client import Config

from api_app import app

LOG_PATH = "test_service.log"
FINAL_KEY = "recsys/recommendations/recommendations.parquet"

# Загрузка переменных окружения из .env файла
load_dotenv()

def _s3():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        endpoint_url="https://storage.yandexcloud.net",
        config=Config(signature_version="s3v4", s3={"payload_signing_enabled": False}),
    )

def _read_parquet(bucket: str, key: str) -> pd.DataFrame:
    s3 = _s3()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))

def _pick_known_user() -> int:
    bucket = os.getenv("student_s3_bucket")
    df = _read_parquet(bucket, FINAL_KEY)
    return int(df["user_id"].iloc[0])

def _pick_unknown_user(known: int) -> int:
    return known + 99_999_999

with TestClient(app) as client, open(LOG_PATH, "w", encoding="utf-8") as fout:
    fout.write("=== Тестирование сервиса рекомендаций ===\n\n")
    
    # 0) Health check
    r = client.get("/health")
    fout.write(f"0. Health check: {r.status_code} {r.json()}\n\n")
    assert r.status_code == 200

    known = _pick_known_user()
    unknown = _pick_unknown_user(known)

    # 1) Пользователь БЕЗ истории (новый) → top_popular
    fout.write("1. Тест: Пользователь БЕЗ истории (новый user_id)\n")
    r = client.post("/recommend", json={"user_id": unknown, "k": 10})
    fout.write(f"   Статус: {r.status_code}\n")
    fout.write(f"   Ответ: {r.text}\n")
    assert r.status_code == 200
    data_no_history = r.json()
    assert data_no_history["user_id"] == unknown
    assert len(data_no_history["tracks"]) == 10
    fout.write(f"   ✅ Получено {len(data_no_history['tracks'])} треков из top_popular\n")
    fout.write(f"   Треки: {data_no_history['tracks'][:5]}...\n\n")

    # 2) Пользователь С историей, БЕЗ онлайн-сигналов → персональные
    fout.write("2. Тест: Пользователь С историей, БЕЗ онлайн-сигналов\n")
    r = client.post("/recommend", json={"user_id": known, "k": 10})
    fout.write(f"   Статус: {r.status_code}\n")
    fout.write(f"   Ответ: {r.text}\n")
    assert r.status_code == 200
    data_with_history = r.json()
    assert data_with_history["user_id"] == known
    assert len(data_with_history["tracks"]) == 10
    fout.write(f"   ✅ Получено {len(data_with_history['tracks'])} персональных треков\n")
    fout.write(f"   Треки: {data_with_history['tracks'][:5]}...\n\n")

    # 3) Пользователь С историей + онлайн-сигналы (recent_tracks)
    fout.write("3. Тест: Пользователь С историей + онлайн-сигналы (recent_tracks)\n")
    recent = data_with_history["tracks"][:3]  # берём первые 3 трека как "недавно прослушанные"
    r = client.post("/recommend", json={
        "user_id": known, 
        "k": 10,
        "recent_tracks": recent
    })
    fout.write(f"   Статус: {r.status_code}\n")
    fout.write(f"   Недавно прослушанные: {recent}\n")
    fout.write(f"   Ответ: {r.text}\n")
    assert r.status_code == 200
    data_with_online = r.json()
    assert data_with_online["user_id"] == known
    assert len(data_with_online["tracks"]) == 10
    fout.write(f"   ✅ Получено {len(data_with_online['tracks'])} треков с учётом онлайн-истории\n")
    fout.write(f"   Треки: {data_with_online['tracks'][:5]}...\n\n")

    # 4) Статистика
    r = client.get("/stats")
    fout.write("4. Статистика работы сервиса:\n")
    fout.write(f"   {json.dumps(r.json(), indent=2, ensure_ascii=False)}\n\n")

    fout.write("=== Все тесты пройдены успешно ===\n")
