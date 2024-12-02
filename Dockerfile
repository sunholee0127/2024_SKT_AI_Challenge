# Python 3 공식 이미지를 기반으로 함
FROM python:3 AS base 
RUN pip install poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \    
    POETRY_VIRTURALENVS_CREATE=true \
    POETRY_CACHE_DIR='/tmp/poetry_cache'

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --without dev --no-root

FROM python:3-slim AS runtime
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"
COPY --from=base ${VIRTUAL_ENV} ${VIRTUAL_ENV}
WORKDIR /app
COPY documents ./documents
#COPY data ./data
COPY tbrand-chatbot ./tbrand-chatbot
COPY tbrandchatbot_banner.png ./tbrandchatbot_banner.png
COPY skt-ai-challenge-51b817bfbe25.json ./skt-ai-challenge-51b817bfbe25.json

#ENTRYPOINT [ "streamlit", "run", "tbrand-chatbot/main.py" ]




