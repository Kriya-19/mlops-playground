import os
import logging

logger = logging.getLogger(__name__)

SNS_ENABLED = os.getenv("SNS_ENABLED", "false").lower() == "true"


async def send_alert(subject: str, message: str) -> bool:
    """
    Send SNS alert. STUB until Step 7.
    Logs locally for now.
    """
    if not SNS_ENABLED:
        logger.info(f"[SNS STUB] Subject: {subject} | Message: {message}")
        return True

    # Full AWS SNS implementation added in Step 7
    raise NotImplementedError("AWS SNS not yet configured.")