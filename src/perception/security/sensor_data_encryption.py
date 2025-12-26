"""
Security module for encrypting sensor data during transmission and storage.
"""
import hashlib
import hmac
import secrets
from typing import Union, Dict, Any, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import json


class SensorDataEncryption:
    """
    Provides encryption and decryption capabilities for sensor data.
    """

    def __init__(self, password: Optional[str] = None):
        """
        Initialize the encryption module.

        Args:
            password: Optional password for encryption key derivation.
                     If None, a random key will be generated.
        """
        if password:
            self.key = self._derive_key_from_password(password)
        else:
            self.key = Fernet.generate_key()

        self.cipher_suite = Fernet(self.key)

    def _derive_key_from_password(self, password: str) -> bytes:
        """
        Derive an encryption key from a password using PBKDF2.

        Args:
            password: Password to derive key from

        Returns:
            bytes: Derived encryption key
        """
        # Use a random salt for key derivation
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def encrypt_sensor_data(self, data: Union[Dict[str, Any], str, bytes]) -> Dict[str, Any]:
        """
        Encrypt sensor data for secure transmission/storage.

        Args:
            data: Sensor data to encrypt (can be dict, string, or bytes)

        Returns:
            Dict[str, Any]: Encrypted data with metadata
        """
        # Convert data to bytes if it's not already
        if isinstance(data, dict):
            data_bytes = json.dumps(data).encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            raise ValueError(f"Unsupported data type for encryption: {type(data)}")

        # Encrypt the data
        encrypted_data = self.cipher_suite.encrypt(data_bytes)

        # Create result with metadata
        result = {
            'encrypted_data': base64.b64encode(encrypted_data).decode('utf-8'),
            'encryption_method': 'Fernet',
            'timestamp': self._get_current_timestamp()
        }

        return result

    def decrypt_sensor_data(self, encrypted_data: Dict[str, Any]) -> Union[Dict[str, Any], str]:
        """
        Decrypt sensor data.

        Args:
            encrypted_data: Dictionary containing encrypted data and metadata

        Returns:
            Union[Dict[str, Any], str]: Decrypted data (dict if original was dict, str otherwise)
        """
        if 'encrypted_data' not in encrypted_data:
            raise ValueError("Missing encrypted_data field in input")

        if encrypted_data.get('encryption_method') != 'Fernet':
            raise ValueError(f"Unsupported encryption method: {encrypted_data.get('encryption_method')}")

        # Decode the encrypted data
        encrypted_bytes = base64.b64decode(encrypted_data['encrypted_data'])

        # Decrypt the data
        decrypted_bytes = self.cipher_suite.decrypt(encrypted_bytes)

        # Try to parse as JSON first, otherwise return as string
        try:
            return json.loads(decrypted_bytes.decode('utf-8'))
        except json.JSONDecodeError:
            return decrypted_bytes.decode('utf-8')

    def generate_access_token(self, sensor_id: str, permissions: str = "read",
                            expiry_hours: int = 24) -> str:
        """
        Generate an access token for sensor data access control.

        Args:
            sensor_id: ID of the sensor
            permissions: Permissions for the token (default: "read")
            expiry_hours: Number of hours until token expires (default: 24)

        Returns:
            str: Generated access token
        """
        import time
        import jwt

        # Create payload with sensor ID, permissions, and expiry
        payload = {
            'sensor_id': sensor_id,
            'permissions': permissions,
            'exp': time.time() + (expiry_hours * 3600),  # Convert hours to seconds
            'iat': time.time()  # Issued at time
        }

        # Sign the token with the encryption key
        token = jwt.encode(payload, self.key, algorithm='HS256')
        return token

    def verify_access_token(self, token: str, required_permissions: str = "read") -> bool:
        """
        Verify an access token for sensor data access.

        Args:
            token: Access token to verify
            required_permissions: Required permissions for access

        Returns:
            bool: True if token is valid and has required permissions, False otherwise
        """
        import jwt
        import time

        try:
            # Decode the token with the encryption key
            payload = jwt.decode(token, self.key, algorithms=['HS256'])

            # Check if token has expired
            if payload['exp'] < time.time():
                return False

            # Check if token has required permissions
            if payload['permissions'] != required_permissions:
                return False

            return True

        except jwt.ExpiredSignatureError:
            return False
        except jwt.InvalidTokenError:
            return False
        except Exception:
            return False

    def hash_sensor_data(self, data: Union[Dict[str, Any], str, bytes]) -> str:
        """
        Generate a hash of sensor data for integrity verification.

        Args:
            data: Sensor data to hash

        Returns:
            str: Hash of the sensor data
        """
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
            data_bytes = data_str.encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            raise ValueError(f"Unsupported data type for hashing: {type(data)}")

        # Create SHA-256 hash
        hash_obj = hashlib.sha256(data_bytes)
        return hash_obj.hexdigest()

    def verify_data_integrity(self, data: Union[Dict[str, Any], str, bytes],
                            expected_hash: str) -> bool:
        """
        Verify the integrity of sensor data against an expected hash.

        Args:
            data: Sensor data to verify
            expected_hash: Expected hash of the data

        Returns:
            bool: True if data integrity is verified, False otherwise
        """
        actual_hash = self.hash_sensor_data(data)
        return hmac.compare_digest(actual_hash, expected_hash)

    def _get_current_timestamp(self) -> float:
        """
        Get the current timestamp.

        Returns:
            float: Current timestamp
        """
        import time
        return time.time()

    def encrypt_sensor_payload(self, sensor_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encrypt a sensor data payload with metadata.

        Args:
            sensor_id: ID of the sensor
            payload: Sensor data payload to encrypt

        Returns:
            Dictionary containing encrypted payload and metadata
        """
        encrypted_result = self.encrypt_sensor_data(payload)

        # Create metadata
        metadata = {
            'sensor_id': sensor_id,
            'encrypted': True,
            'encryption_method': 'fernet',
            'timestamp': payload.get('timestamp', ''),
            'data_hash': self.hash_sensor_data(payload)
        }

        return {
            'encrypted_payload': encrypted_result['encrypted_data'],
            'metadata': metadata
        }

    def decrypt_sensor_payload(self, encrypted_package: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Decrypt a sensor data payload.

        Args:
            encrypted_package: Dictionary containing encrypted payload and metadata

        Returns:
            Decrypted payload or None if decryption fails
        """
        if not encrypted_package.get('encrypted_payload'):
            return None

        # Create a dictionary in the expected format for decrypt_sensor_data
        encrypted_data_dict = {
            'encrypted_data': encrypted_package['encrypted_payload'],
            'encryption_method': 'Fernet'
        }

        # Decrypt the data
        decrypted_data = self.decrypt_sensor_data(encrypted_data_dict)

        if decrypted_data is None:
            return None

        return decrypted_data

    def generate_checksum(self, data: Union[Dict[str, Any], str, bytes]) -> str:
        """
        Generate a checksum for data integrity verification.

        Args:
            data: Data to generate checksum for

        Returns:
            SHA-256 checksum as hex string
        """
        return self.hash_sensor_data(data)

    def verify_checksum(self, data: Union[Dict[str, Any], str, bytes], expected_checksum: str) -> bool:
        """
        Verify data integrity against a checksum.

        Args:
            data: Data to verify
            expected_checksum: Expected checksum

        Returns:
            True if checksum matches, False otherwise
        """
        return self.verify_data_integrity(data, expected_checksum)


def create_sensor_data_signature(data: Union[str, Dict[str, Any]], private_key: Optional[str] = None) -> str:
    """
    Create a digital signature for sensor data (simplified implementation).

    Args:
        data: Data to sign
        private_key: Private key for signing (if None, uses a default)

    Returns:
        Signature string
    """
    import hashlib
    import json

    if isinstance(data, dict):
        data_str = json.dumps(data, sort_keys=True)
    else:
        data_str = str(data)

    # In a real implementation, this would use proper digital signature algorithms
    # For now, we'll create a hash-based signature
    signature = hashlib.sha256(data_str.encode('utf-8')).hexdigest()
    return signature


def verify_sensor_data_signature(data: Union[str, Dict[str, Any]], signature: str, public_key: Optional[str] = None) -> bool:
    """
    Verify a digital signature for sensor data (simplified implementation).

    Args:
        data: Data that was signed
        signature: Signature to verify
        public_key: Public key for verification (if None, uses a default)

    Returns:
        True if signature is valid, False otherwise
    """
    expected_signature = create_sensor_data_signature(data, public_key)
    return expected_signature == signature


# Example usage and testing
if __name__ == "__main__":
    # Create an encryption instance
    encryptor = SensorDataEncryption(password="my_secure_password")

    # Example sensor data
    sensor_data = {
        "sensor_id": "camera_front",
        "timestamp": 1234567890.123,
        "data": [1, 2, 3, 4, 5],
        "metadata": {"resolution": "1920x1080", "format": "RGB"}
    }

    # Encrypt the data
    encrypted_result = encryptor.encrypt_sensor_data(sensor_data)
    print(f"Encrypted data: {encrypted_result}")

    # Decrypt the data
    decrypted_data = encryptor.decrypt_sensor_data(encrypted_result)
    print(f"Decrypted data: {decrypted_data}")

    # Verify that the original and decrypted data match
    print(f"Data matches: {sensor_data == decrypted_data}")

    # Generate and verify an access token
    token = encryptor.generate_access_token("camera_front", "read", 1)  # 1 hour expiry
    print(f"Generated token: {token}")
    print(f"Token valid: {encryptor.verify_access_token(token)}")

    # Hash the data for integrity verification
    data_hash = encryptor.hash_sensor_data(sensor_data)
    print(f"Data hash: {data_hash}")
    print(f"Integrity check: {encryptor.verify_data_integrity(sensor_data, data_hash)}")

    # Test the new payload encryption/decryption
    encrypted_payload = encryptor.encrypt_sensor_payload("camera_front", sensor_data)
    print(f"Encrypted payload: {encrypted_payload}")

    decrypted_payload = encryptor.decrypt_sensor_payload(encrypted_payload)
    print(f"Decrypted payload: {decrypted_payload}")
    print(f"Payload matches: {sensor_data == decrypted_payload}")