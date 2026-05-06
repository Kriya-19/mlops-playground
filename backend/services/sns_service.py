import os
import asyncio
import logging

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

SNS_ENABLED = os.getenv("SNS_ENABLED", "false").lower() == "true"
AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
SNS_TOPIC_ARN = os.getenv("SNS_TOPIC_ARN")


async def send_alert(subject: str, message: str) -> bool:
    """
    Send an SNS alert to the configured topic.

    Falls back to a log-only stub when SNS is disabled or the topic ARN is
    unavailable. Returns True when the message is published or intentionally
    skipped.
    """
    if not SNS_ENABLED or not SNS_TOPIC_ARN:
        logger.info(f"[SNS STUB] Subject: {subject} | Message: {message}")
        return True

    def _publish() -> str:
        client = boto3.client("sns", region_name=AWS_REGION)
        response = client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject[:100],
            Message=message,
        )
        return response.get("MessageId", "")

    try:
        message_id = await asyncio.to_thread(_publish)
        logger.info(f"✅ SNS notification sent | MessageId={message_id}")
        return True
    except (BotoCoreError, ClientError, Exception) as exc:
        logger.warning(f"SNS publish failed: {exc}")
        return False