from typing import Annotated

from fastapi import Depends

from app.services.chat_service import ChatService, get_chat_service
from app.services.ingest_service import IngestService, get_ingest_service

# Annotated aliases: routes just declare `service: ChatServiceDep`.
# In tests: app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
IngestServiceDep = Annotated[IngestService, Depends(get_ingest_service)]