# server.py
"""
FastAPI сервер для классификации страйкбольного снаряжения.
Поддерживает множественные товары в одном объявлении и уточнение подкатегорий через текст.
"""
import os
import uvicorn
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import FastAPI, HTTPException, Depends, Header, File, UploadFile, Security
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from contextlib import asynccontextmanager
from ultralytics import YOLO
import requests
import torch
import logging
from pathlib import Path
from io import BytesIO
from PIL import Image
import urllib3

# Отключаем предупреждения SSL (только для локальных тестов)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === КОНФИГУРАЦИЯ ===
MODEL_PATH = "./runs/classify/runs/airsoft_cls_v1/weights/best.pt"

# Читаем ключ из переменной окружения. Если её нет — берем значение по умолчанию.
API_KEY = os.getenv("API_KEY", "my_super_secret_key_12345") 

# 1. Маппинг визуальных классов YOLO в общие категории
VISUAL_PARENTS = {
    "ak": "Оружие", "m_series": "Оружие", "hk": "Оружие", 
    "rifle": "Оружие", "pistol": "Оружие", "shotgun": "Оружие", "machinegun": "Оружие",
    "vest": "Экипировка", "helmet": "Защита", 
    "backpack": "Аксессуары", "pouch": "Аксессуары"
}

# 2. Паттерны для анализа текста (иерархическая структура)
# parent должен совпадать с ключом из VISUAL_PARENTS
TEXT_PATTERNS = [
    # --- Оружие ---
    {"keywords": ["винтовка", "снайперка", "дрегунов", "cm708", "cyma 708"], "sub": "Снайперские винтовки", "parent": "rifle"},
    {"keywords": ["пистолет", "пистоль", "glock", "1911", "hi-capa"], "sub": "Пистолеты", "parent": "pistol"},
    {"keywords": ["калаш", "ак-74", "akm", "айрон", "ak-12", "ак-74м"], "sub": "Автоматы (AK серия)", "parent": "ak"},
    {"keywords": ["м4", "m4", "hk416", "hk", "ар-15", "m-series"], "sub": "Автоматы (M/HK серия)", "parent": "m_series"},
    {"keywords": ["дробовик", "шотган", "m870", "spas"], "sub": "Дробовики", "parent": "shotgun"},
    {"keywords": ["пулемет", "minimi", "m249", "пкп", "pecheneg"], "sub": "Пулеметы", "parent": "machinegun"},

    # --- Экипировка (ранее папка vest) ---
    {"keywords": ["жилет", "разгрузка", "plate carrier", "панцирь", "жилетка", "жилет брон"], "sub": "Тактические жилеты", "parent": "vest"},
    {"keywords": ["пояс", "battle belt", "поясная система", "ремень боевой", "пояс тактический"], "sub": "Боевые пояса", "parent": "vest"},
    {"keywords": ["нагрудник", "chest rig", "нагрудная разгрузка", "нагрудник тактический"], "sub": "Нагрудники", "parent": "vest"},
    {"keywords": ["набедренная", "thigh rig", "платформа бедренная", "бедро", "набедренник"], "sub": "Платформы набедренные", "parent": "vest"},
    {"keywords": ["бандольера", "плечевая разгрузка", "патронташ плечевой"], "sub": "Бандольеры", "parent": "vest"},

    # --- Аксессуары ---
    {"keywords": ["подсумок", "pouch", "935", "eagle", "айгл", "под магазин"], "sub": "Подсумки", "parent": "pouch"},
    {"keywords": ["двойной под", "double mag", "под два магазина", "сдвоенный"], "sub": "Подсумки (Double)", "parent": "pouch"},
    {"keywords": ["рюкзак", "баул", "сумка тактическая", "backpack", "ранец"], "sub": "Рюкзаки и сумки", "parent": "backpack"},

    # --- Защита ---
    {"keywords": ["шлем", "маска", "helmet", "голова", "full face"], "sub": "Шлемы и маски", "parent": "helmet"},
]

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка модели при старте"""
    global model
    if not Path(MODEL_PATH).exists():
        raise RuntimeError(f"❌ Модель не найдена: {MODEL_PATH}")
    
    logger.info("🔄 Загрузка модели YOLO...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(MODEL_PATH).to(device)
    logger.info(f"✅ Модель загружена на устройство: {device}")
    yield

app = FastAPI(
    title="Airsoft Classification API",
    description="API для классификации страйкбольного снаряжения по тексту и фото",
    version="1.0.0",
    lifespan=lifespan,
    #  Добавляем описание схем безопасности для Swagger UI
    openapi_tags=[
        {"name": "Prediction", "description": "Эндпоинты для получения предсказаний"}
    ],
    servers=[{"url": "/", "description": "Local server"}]
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# === АУТЕНТИФИКАЦИЯ ===
security = HTTPBearer(auto_error=False)

# Функция проверки ключа
async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> str:
    """
    Проверяет API-ключ из заголовка Authorization: Bearer <key>
    """
    if credentials is None or credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=401, 
            detail="Неверный API ключ. Передайте ключ в формате: Authorization: Bearer <ваш_ключ>"
        )
    return credentials.credentials

# === СХЕМЫ ДАННЫХ ===
class PhotoInput(BaseModel):
    photo_id: str
    url: HttpUrl

class PredictionRequest(BaseModel):
    post_id: str
    text: str
    photos: List[PhotoInput]

class PredictionItem(BaseModel):
    object_id: str
    category: str
    subcategory: str
    confidence: float
    source: str  # 'text', 'image', или 'text+image'
    photo_id: Optional[str] = None

class PredictionResponse(BaseModel):
    post_id: str
    predictions: List[PredictionItem]

# === ЛОГИКА ОБРАБОТКИ ===

def analyze_text(text: str) -> List[dict]:
    """
    Сканирует текст и возвращает список найденных товаров.
    Поддерживает несколько товаров одной категории (например, 2 вида подсумков).
    """
    found_items = []
    text_lower = text.lower()
    
    for pattern in TEXT_PATTERNS:
        # Проверяем, есть ли хоть одно ключевое слово паттерна в тексте
        if any(kw in text_lower for kw in pattern["keywords"]):
            found_items.append({
                "category": VISUAL_PARENTS.get(pattern["parent"], "Другое"),
                "subcategory": pattern["sub"],
                "parent_class": pattern["parent"],
                "confidence": 0.75,
                "source": "text",
                "object_id": f"text_{pattern['sub'].replace(' ', '_')}"
            })
    return found_items

def analyze_image(photo: PhotoInput) -> Optional[dict]:
    """Скачивает фото и классифицирует через YOLO"""
    try:
        logger.info(f" Скачивание: {photo.url}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(str(photo.url), headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content)).convert('RGB')
        results = model.predict(source=img, verbose=False)
        
        top_idx = results[0].probs.top1
        confidence = results[0].probs.data[top_idx].item()
        class_name = model.names[top_idx]
        
        if confidence < 0.5:
            return None
            
        category = VISUAL_PARENTS.get(class_name, "Неизвестно")
        # По умолчанию подкатегория = название класса, текст её уточнит позже
        default_sub = class_name.replace('_', ' ').title()
        
        return {
            "category": category,
            "subcategory": default_sub,
            "parent_class": class_name,
            "confidence": round(confidence, 3),
            "source": "image",
            "object_id": f"img_{photo.photo_id}",
            "photo_id": photo.photo_id
        }
    except Exception as e:
        logger.error(f"❌ Ошибка фото {photo.photo_id}: {e}")
        return None

def merge_predictions(text_preds: List[dict], image_preds: List[dict]) -> List[PredictionItem]:
    """
    Объединяет результаты текста и фото.
    Логика: Фото подтверждает общую категорию (parent). Текст даёт точное название (sub).
    Если текст нашёл несколько товаров одной группы — вернёт все.
    """
    final = {}
    
    # 1. Добавляем визуальные находки
    for p in image_preds:
        key = (p["parent_class"], p["subcategory"])
        if key not in final or p["confidence"] > final[key].confidence:
            final[key] = p

    # 2. Накладываем текстовые находки
    for p in text_preds:
        parent = p["parent_class"]
        sub = p["subcategory"]
        key = (parent, sub)
        
        if key in final:
            # Фото уже нашло эту группу. Обновляем подкатегорию на текстовую и помечаем как комбо
            existing = final[key]
            existing["subcategory"] = sub
            existing["confidence"] = round((existing["confidence"] + p["confidence"]) / 2, 3)
            existing["source"] = "text+image"
        else:
            # Текст нашёл что-то, чего нет на фото (или фото не загрузилось)
            final[key] = p

    # Преобразуем в Pydantic объекты
    return [
        PredictionItem(
            object_id=p["object_id"],
            category=p["category"],
            subcategory=p["subcategory"],
            confidence=p["confidence"],
            source=p["source"],
            photo_id=p.get("photo_id")
        )
        for p in final.values()
    ]

# === ЭНДПОИНТЫ ===

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    request: PredictionRequest,
    api_key: str = Security(verify_api_key) 
):
    logger.info(f"Запрос: {request.post_id}")
    
    t_preds = analyze_text(request.text)
    i_preds = [p for photo in request.photos if (p := analyze_image(photo))]
    
    return PredictionResponse(
        post_id=request.post_id,
        predictions=merge_predictions(t_preds, i_preds)
    )

@app.post("/predict-local", response_model=PredictionResponse, tags=["Prediction"])
async def predict_local(
    post_id: str,
    text: str,
    photo: UploadFile = File(...),
    api_key: str = Security(verify_api_key)  
):
    logger.info(f"Локальный: {post_id}")
    
    t_preds = analyze_text(text)
    i_preds = []
    try:
        img = Image.open(photo.file).convert('RGB')
        results = model.predict(source=img, verbose=False)
        top_idx = results[0].probs.top1
        conf = results[0].probs.data[top_idx].item()
        cls = model.names[top_idx]
        
        if conf >= 0.5:
            i_preds.append({
                "category": VISUAL_PARENTS.get(cls, "Неизвестно"),
                "subcategory": cls.replace('_', ' ').title(),
                "parent_class": cls,
                "confidence": round(conf, 3),
                "source": "image",
                "object_id": "img_local",
                "photo_id": "local"
            })
    except Exception as e:
        logger.error(f"❌ Ошибка файла: {e}")
        
    return PredictionResponse(
        post_id=post_id,
        predictions=merge_predictions(t_preds, i_preds)
    )

@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)