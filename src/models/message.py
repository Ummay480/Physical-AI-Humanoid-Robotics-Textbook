"""
Message models for inter-agent communication.

This module defines message types and structures for communication
between subagents in the framework.
"""

from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import UUID, uuid4


class MessageType(str, Enum):
    """
    Types of inter-agent messages.

    Message Types:
        EVENT: Notification of an event (fire-and-forget)
        REQUEST: Request for action or data (expects response)
        RESPONSE: Response to a previous request
        ERROR: Error notification
        HEARTBEAT: Keep-alive/health check message
        CONTROL: System control message (pause, resume, cancel)
    """

    EVENT = "event"
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    CONTROL = "control"

    @property
    def expects_response(self) -> bool:
        """Check if this message type expects a response."""
        return self == MessageType.REQUEST

    @property
    def is_error(self) -> bool:
        """Check if this is an error message."""
        return self == MessageType.ERROR


class MessagePriority(str, Enum):
    """
    Message priority levels for routing and processing.

    Priority Levels:
        LOW: Background tasks, non-urgent notifications
        NORMAL: Standard operations
        HIGH: Important tasks requiring prompt attention
        CRITICAL: Emergency situations, system failures
    """

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

    def __lt__(self, other):
        """Compare priorities for sorting."""
        priority_order = {
            MessagePriority.LOW: 0,
            MessagePriority.NORMAL: 1,
            MessagePriority.HIGH: 2,
            MessagePriority.CRITICAL: 3,
        }
        return priority_order[self] < priority_order[other]


class MessageHeader(BaseModel):
    """Message header containing routing and metadata information."""

    message_id: UUID = Field(default_factory=uuid4, description="Unique message identifier")
    message_type: MessageType = Field(..., description="Type of message")
    sender_id: UUID = Field(..., description="ID of sending agent")
    recipient_id: Optional[UUID] = Field(None, description="ID of recipient agent (None for broadcast)")
    topic: Optional[str] = Field(None, description="Topic for pub/sub messaging")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message creation time")
    priority: MessagePriority = Field(default=MessagePriority.NORMAL, description="Message priority")
    correlation_id: Optional[UUID] = Field(None, description="ID of related message (for request/response)")
    ttl_seconds: Optional[float] = Field(None, description="Time-to-live in seconds")

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }


class MessagePayload(BaseModel):
    """Message payload containing the actual data."""

    data: Dict[str, Any] = Field(default_factory=dict, description="Message data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from data with default."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in data."""
        self.data[key] = value


class Message(BaseModel):
    """
    Inter-agent message for communication.

    Messages enable subagents to communicate events, requests, responses,
    and errors. They support both point-to-point and topic-based messaging.

    Attributes:
        header: Routing and metadata information
        payload: Message data and content
    """

    header: MessageHeader
    payload: MessagePayload = Field(default_factory=MessagePayload)

    @property
    def message_id(self) -> UUID:
        """Get message ID."""
        return self.header.message_id

    @property
    def message_type(self) -> MessageType:
        """Get message type."""
        return self.header.message_type

    @property
    def sender_id(self) -> UUID:
        """Get sender agent ID."""
        return self.header.sender_id

    @property
    def recipient_id(self) -> Optional[UUID]:
        """Get recipient agent ID."""
        return self.header.recipient_id

    @property
    def topic(self) -> Optional[str]:
        """Get message topic."""
        return self.header.topic

    @property
    def is_broadcast(self) -> bool:
        """Check if this is a broadcast message (no specific recipient)."""
        return self.header.recipient_id is None and self.header.topic is not None

    @property
    def is_expired(self) -> bool:
        """Check if message has exceeded its TTL."""
        if self.header.ttl_seconds is None:
            return False

        age_seconds = (datetime.utcnow() - self.header.timestamp).total_seconds()
        return age_seconds > self.header.ttl_seconds

    def create_response(self, data: Dict[str, Any]) -> "Message":
        """
        Create a response message to this message.

        Args:
            data: Response data

        Returns:
            Response message with correlation ID set

        Raises:
            ValueError: If this message is not a REQUEST type
        """
        if not self.header.message_type.expects_response:
            raise ValueError(f"Cannot create response for message type: {self.header.message_type}")

        response_header = MessageHeader(
            message_type=MessageType.RESPONSE,
            sender_id=self.header.recipient_id or uuid4(),  # Swap sender/recipient
            recipient_id=self.header.sender_id,
            correlation_id=self.header.message_id,
            priority=self.header.priority,
        )

        return Message(
            header=response_header,
            payload=MessagePayload(data=data),
        )

    def create_error_response(self, error_message: str, error_code: Optional[str] = None) -> "Message":
        """
        Create an error response message.

        Args:
            error_message: Error description
            error_code: Optional error code

        Returns:
            Error message with correlation ID set
        """
        error_header = MessageHeader(
            message_type=MessageType.ERROR,
            sender_id=self.header.recipient_id or uuid4(),
            recipient_id=self.header.sender_id,
            correlation_id=self.header.message_id,
            priority=MessagePriority.HIGH,
        )

        error_data = {
            "error_message": error_message,
            "error_code": error_code,
            "original_message_id": str(self.header.message_id),
        }

        return Message(
            header=error_header,
            payload=MessagePayload(data=error_data),
        )

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
            MessageType: lambda v: v.value,
            MessagePriority: lambda v: v.value,
        }


# Convenience factory functions

def create_event_message(
    sender_id: UUID,
    topic: str,
    data: Dict[str, Any],
    priority: MessagePriority = MessagePriority.NORMAL,
) -> Message:
    """Create an event message for topic-based broadcasting."""
    header = MessageHeader(
        message_type=MessageType.EVENT,
        sender_id=sender_id,
        topic=topic,
        priority=priority,
    )
    return Message(header=header, payload=MessagePayload(data=data))


def create_request_message(
    sender_id: UUID,
    recipient_id: UUID,
    data: Dict[str, Any],
    priority: MessagePriority = MessagePriority.NORMAL,
    ttl_seconds: Optional[float] = None,
) -> Message:
    """Create a request message for point-to-point communication."""
    header = MessageHeader(
        message_type=MessageType.REQUEST,
        sender_id=sender_id,
        recipient_id=recipient_id,
        priority=priority,
        ttl_seconds=ttl_seconds,
    )
    return Message(header=header, payload=MessagePayload(data=data))


def create_heartbeat_message(sender_id: UUID, status_data: Optional[Dict[str, Any]] = None) -> Message:
    """Create a heartbeat message for health monitoring."""
    header = MessageHeader(
        message_type=MessageType.HEARTBEAT,
        sender_id=sender_id,
        priority=MessagePriority.LOW,
    )
    return Message(header=header, payload=MessagePayload(data=status_data or {}))


def create_control_message(
    sender_id: UUID,
    recipient_id: UUID,
    command: str,
    parameters: Optional[Dict[str, Any]] = None,
) -> Message:
    """Create a control message for system commands (pause, resume, cancel)."""
    header = MessageHeader(
        message_type=MessageType.CONTROL,
        sender_id=sender_id,
        recipient_id=recipient_id,
        priority=MessagePriority.HIGH,
    )
    data = {"command": command, "parameters": parameters or {}}
    return Message(header=header, payload=MessagePayload(data=data))
