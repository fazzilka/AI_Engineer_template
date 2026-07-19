# AI Engineer Template

[![CI](https://github.com/fazzilka/AI_Engineer_template/actions/workflows/ci.yml/badge.svg)](https://github.com/fazzilka/AI_Engineer_template/actions/workflows/ci.yml)

Универсальный production-oriented шаблон для AI-приложений на Python. Он даёт разработчику рабочий
вертикальный срез — от HTTP-контракта до LLM-провайдера — и оставляет продуктовые решения вроде RAG,
агентов, очередей и конкретной базы данных подключаемыми модулями.

Шаблон подходит как основа для AI API, copilots, внутренних ассистентов, RAG-сервисов, agentic
workflows и фоновых AI-задач.

## Что уже есть

- FastAPI с версионированным chat endpoint и OpenAPI-документацией.
- Provider-neutral `LLMClient` и адаптер OpenAI-compatible Chat Completions API.
- Offline fake-провайдер: первый запуск, тесты и CI не требуют ключей или сети.
- Версионируемый system prompt как часть исходного кода.
- Bounded retries, timeout и безопасный публичный `503` при сбое провайдера.
- Структурные логи, request ID, Prometheus-метрики HTTP/LLM latency и token usage.
- Лёгкий eval harness с JSONL-набором регрессионных сценариев.
- Строгие quality gates: Ruff, mypy, pytest, branch coverage не ниже 90%.
- `uv.lock`, Makefile, hardened Docker image, Docker Compose, GitHub Actions и Dependabot.
- `AGENTS.md` с правилами для Codex и других coding agents.

В шаблоне намеренно нет обязательных Postgres, Redis, vector DB или agent framework. Они полезны не
каждому AI-продукту и должны подключаться через узкие порты, когда появляется реальная задача.

## Быстрый старт

Понадобятся Python 3.12+, [uv](https://docs.astral.sh/uv/) 0.11.x и Make.

```bash
cp .env.example .env
make install
make check
make dev
```

Сервис запустится на `http://localhost:8000`, Swagger UI — на `http://localhost:8000/docs`.
По умолчанию используется детерминированный fake-провайдер, поэтому следующий запрос работает сразу:

```bash
curl --request POST http://localhost:8000/api/v1/chat \
  --header 'Content-Type: application/json' \
  --header 'X-Request-ID: local-example' \
  --data '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

Пример ответа:

```json
{
  "content": "Fake response: Hello!",
  "model": "fake-model",
  "finish_reason": "stop",
  "usage": {
    "input_tokens": 1,
    "output_tokens": 2
  }
}
```

## Подключение реальной модели

Измените `.env`:

```dotenv
LLM__PROVIDER=openai_compatible
LLM__MODEL=your-model-name
LLM__API_KEY=your-api-key
```

Для OpenAI-compatible сервера с собственным endpoint добавьте:

```dotenv
LLM__BASE_URL=http://localhost:11434/v1
LLM__API_KEY=local
```

Модель выбирается только серверной конфигурацией: клиент API не может подменить provider, base URL или
credentials. Это уменьшает поверхность SSRF, утечки ключей и неконтролируемого расхода токенов.

## API

| Endpoint | Назначение |
| --- | --- |
| `POST /api/v1/chat` | Получить ответ модели на историю сообщений |
| `GET /health/live` | Проверить, что процесс жив |
| `GET /health/ready` | Проверить инициализацию приложения |
| `GET /metrics` | Получить Prometheus-метрики |
| `GET /docs` | Открыть Swagger UI, если `DOCS_ENABLED=true` |

Chat API принимает до 100 сообщений с ролями `user` и `assistant`. Последнее сообщение обязательно
должно принадлежать пользователю. System prompt хранится и контролируется на сервере.

## Evals

`evals/cases.jsonl` — минимальный version-controlled набор AI-регрессий. Каждый сценарий задаёт историю
сообщений и обязательные/запрещённые фрагменты ответа.

```bash
make eval
uv run python -m evals.run --dataset evals/cases.jsonl --min-pass-rate 0.9
```

Стартовые кейсы показывают формат, но не заменяют продуктовый eval suite. При изменении prompt,
retrieval, tools или model parameters добавляйте случаи из реальных пользовательских сценариев и
фиксируйте ожидаемый pass rate в CI.

## Основные команды

| Команда | Действие |
| --- | --- |
| `make help` | Показать все команды |
| `make install` | Установить зависимости строго из lock-файла |
| `make dev` | Запустить API с auto-reload |
| `make run` | Запустить API без auto-reload |
| `make format` | Отформатировать и безопасно исправить lint-ошибки |
| `make check` | Запустить lint, mypy, тесты с coverage и evals |
| `make build` | Собрать wheel и source distribution |
| `make docker-build` | Собрать production image |
| `make up` / `make down` | Запустить или остановить Docker Compose |

## Структура

```text
src/app/
├── adapters/llm/       # fake и OpenAI-compatible реализации LLMClient
├── api/                # HTTP schemas, dependencies и routes
├── application/        # use cases, не зависящие от FastAPI и provider SDK
├── domain/             # стабильные модели предметной области
├── observability/      # structured logging, metrics, request context
├── ports/              # контракты внешних возможностей
├── prompts/            # versioned prompt assets
├── config.py           # типизированная конфигурация окружения
└── main.py             # composition root и lifecycle
evals/                  # AI regression dataset и runner
tests/                  # unit, adapter contract и API tests
docs/                   # архитектурные решения и extension recipes
```

Поток зависимостей и рецепты расширения описаны в [docs/architecture.md](docs/architecture.md).

## Конфигурация

Настройки читаются из переменных окружения и локального `.env`. Вложенные поля разделяются `__`.

| Переменная | Default | Описание |
| --- | --- | --- |
| `APP_NAME` | `AI Engineer Template` | Имя в OpenAPI |
| `APP_ENV` | `local` | `local`, `test`, `staging` или `production` |
| `LOG_LEVEL` | `INFO` | Уровень логирования |
| `API_PREFIX` | `/api/v1` | Префикс продуктового API |
| `DOCS_ENABLED` | `true` | Публиковать Swagger UI |
| `LLM__PROVIDER` | `fake` | `fake` или `openai_compatible` |
| `LLM__MODEL` | `fake-model` | Серверная модель |
| `LLM__API_KEY` | — | Ключ провайдера; обязателен для remote adapter |
| `LLM__BASE_URL` | provider default | Альтернативный OpenAI-compatible endpoint |
| `LLM__TEMPERATURE` | `0.2` | Температура генерации |
| `LLM__MAX_TOKENS` | `1024` | Максимум output tokens |
| `LLM__TIMEOUT_SECONDS` | `30` | Timeout одного provider request |
| `LLM__MAX_RETRIES` | `2` | Число повторов transient errors |

Никогда не коммитьте `.env`. Для production передавайте секреты через secret manager платформы.

## Docker

```bash
cp .env.example .env
make up
docker compose ps
make logs
```

Runtime image запускает приложение непривилегированным пользователем. Compose дополнительно включает
read-only filesystem, `no-new-privileges` и удаляет Linux capabilities. Перед production-развёртыванием
добавьте TLS, authentication, rate limits и контроль доступа к `/metrics` на ingress/API gateway.

## Использование как GitHub template

Владелец репозитория один раз включает **Settings → General → Template repository**. После этого новый
проект создаётся кнопкой **Use this template**.

После создания проекта:

1. Обновите `name`, `description` и версию в `pyproject.toml`.
2. Настройте `APP_NAME` и LLM provider в `.env`/deployment secrets.
3. Замените `src/app/prompts/system.md` продуктовым prompt.
4. Замените демонстрационные eval cases реальными golden scenarios.
5. Добавьте нужные adapters: persistence, retrieval, tools, queues.
6. Выберите лицензию, security contact и deployment policy своего проекта.

Правила разработки и commit convention находятся в [CONTRIBUTING.md](CONTRIBUTING.md). Правила для
coding agents — в [AGENTS.md](AGENTS.md).
