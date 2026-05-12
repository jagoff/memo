"""Tests for encryption module."""

import pytest

from memo.encryption import (
    Encryptor,
    EncryptionManager,
    EncryptionMetadata,
    KeyDerivation,
    KeyManager,
)


@pytest.fixture
def key_manager(tmp_cfg):
    """Fixture providing KeyManager instance."""
    return KeyManager(tmp_cfg.state_dir)


@pytest.fixture
def encryptor(key_manager):
    """Fixture providing Encryptor instance."""
    return Encryptor(key_manager)


@pytest.fixture
def encryption_manager(key_manager, encryptor):
    """Fixture providing EncryptionManager instance."""
    return EncryptionManager(key_manager, encryptor)


def test_key_manager_init(key_manager):
    """Test KeyManager initialization."""
    assert key_manager.state_dir.is_dir()


def test_key_manager_derive_key(key_manager):
    """Test key derivation from password."""
    key, salt, kd = key_manager.derive_key("testpassword")

    assert len(key) == 32  # 256 bits
    assert len(salt) == 32
    assert kd.iterations == 100000


def test_key_manager_save_key_derivation(key_manager):
    """Test saving key derivation parameters."""
    key, salt, kd = key_manager.derive_key("testpassword")
    key_manager.save_key_derivation(kd)

    loaded = key_manager.load_key_derivation()
    assert loaded is not None
    assert loaded.salt == kd.salt
    assert loaded.iterations == kd.iterations


def test_key_manager_load_key_derivation_none(key_manager):
    """Test loading when no key derivation exists."""
    loaded = key_manager.load_key_derivation()
    assert loaded is None


def test_key_manager_set_master_key(key_manager):
    """Test setting master key."""
    key, salt, kd = key_manager.derive_key("testpassword")
    key_manager.set_master_key(key)

    assert key_manager.get_master_key() == key


def test_key_manager_clear_master_key(key_manager):
    """Test clearing master key."""
    key, salt, kd = key_manager.derive_key("testpassword")
    key_manager.set_master_key(key)
    key_manager.clear_master_key()

    assert key_manager.get_master_key() is None


def test_key_manager_persistence(tmp_cfg):
    """Test key derivation persistence across instances."""
    state_dir = tmp_cfg.state_dir

    # Create first instance and save key derivation
    km1 = KeyManager(state_dir)
    key, salt, kd = km1.derive_key("testpassword")
    km1.save_key_derivation(kd)

    # Create second instance and verify persistence
    km2 = KeyManager(state_dir)
    loaded = km2.load_key_derivation()

    assert loaded is not None
    assert loaded.salt == kd.salt


def test_encryptor_init(encryptor):
    """Test Encryptor initialization."""
    assert encryptor.key_manager is not None


def test_encryptor_encrypt_decrypt(encryptor, key_manager):
    """Test encryption and decryption."""
    # Set master key
    key, salt, kd = key_manager.derive_key("testpassword")
    key_manager.set_master_key(key)

    plaintext = "This is a secret message"
    ciphertext, metadata = encryptor.encrypt(plaintext)

    assert ciphertext
    assert metadata.nonce
    assert metadata.auth_tag

    decrypted = encryptor.decrypt(ciphertext, metadata)
    assert decrypted == plaintext


def test_encryptor_encrypt_no_key(encryptor):
    """Test encryption without master key set."""
    with pytest.raises(ValueError, match="Master key not set"):
        encryptor.encrypt("test")


def test_encryptor_decrypt_no_key(encryptor):
    """Test decryption without master key set."""
    metadata = EncryptionMetadata(
        version="1.0",
        nonce="123",
        auth_tag="456",
    )
    with pytest.raises(ValueError, match="Master key not set"):
        encryptor.decrypt("ciphertext", metadata)


def test_encryptor_encrypt_file(tmp_path, encryptor, key_manager):
    """Test file encryption."""
    # Set master key
    key, salt, kd = key_manager.derive_key("testpassword")
    key_manager.set_master_key(key)

    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Secret content", encoding="utf-8")

    metadata = encryptor.encrypt_file(test_file)

    # Verify file has encryption header
    content = test_file.read_text(encoding="utf-8")
    assert content.startswith("ENC::")
    assert metadata.nonce


def test_encryptor_decrypt_file(tmp_path, encryptor, key_manager):
    """Test file decryption."""
    # Set master key
    key, salt, kd = key_manager.derive_key("testpassword")
    key_manager.set_master_key(key)

    # Create and encrypt test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Secret content", encoding="utf-8")
    encryptor.encrypt_file(test_file)

    # Decrypt
    plaintext, metadata = encryptor.decrypt_file(test_file)

    assert plaintext == "Secret content"
    assert metadata.nonce


def test_encryptor_decrypt_non_encrypted_file(tmp_path, encryptor):
    """Test decrypting non-encrypted file."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Plain text", encoding="utf-8")

    with pytest.raises(ValueError, match="Not an encrypted file"):
        encryptor.decrypt_file(test_file)


def test_encryption_manager_init(encryption_manager):
    """Test EncryptionManager initialization."""
    assert encryption_manager.key_manager is not None
    assert encryption_manager.encryptor is not None


def test_encryption_manager_unlock(encryption_manager):
    """Test unlocking vault."""
    success = encryption_manager.unlock("testpassword")

    assert success is True
    assert encryption_manager.is_unlocked()


def test_encryption_manager_lock(encryption_manager):
    """Test locking vault."""
    encryption_manager.unlock("testpassword")
    encryption_manager.lock()

    assert not encryption_manager.is_unlocked()


def test_encryption_manager_is_unlocked(encryption_manager):
    """Test checking unlock status."""
    assert not encryption_manager.is_unlocked()

    encryption_manager.unlock("testpassword")
    assert encryption_manager.is_unlocked()


def test_encryption_manager_encrypt_memoria(encryption_manager):
    """Test encrypting memoria content."""
    encryption_manager.unlock("testpassword")

    ciphertext, metadata = encryption_manager.encrypt_memoria("id1", "Secret content")

    assert ciphertext
    assert metadata.nonce


def test_encryption_manager_decrypt_memoria(encryption_manager):
    """Test decrypting memoria content."""
    encryption_manager.unlock("testpassword")

    ciphertext, metadata = encryption_manager.encrypt_memoria("id1", "Secret content")
    decrypted = encryption_manager.decrypt_memoria(ciphertext, metadata)

    assert decrypted == "Secret content"


def test_encryption_manager_encrypt_locked(encryption_manager):
    """Test encrypting when vault is locked."""
    with pytest.raises(ValueError, match="Vault is locked"):
        encryption_manager.encrypt_memoria("id1", "test")


def test_encryption_manager_decrypt_locked(encryption_manager):
    """Test decrypting when vault is locked."""
    metadata = EncryptionMetadata(version="1.0", nonce="123", auth_tag="456")

    with pytest.raises(ValueError, match="Vault is locked"):
        encryption_manager.decrypt_memoria("ciphertext", metadata)


def test_encryption_metadata_dataclass():
    """Test EncryptionMetadata dataclass structure."""
    metadata = EncryptionMetadata(
        version="1.0",
        nonce="abc123",
        auth_tag="def456",
        algorithm="AES-256-GCM",
    )
    assert metadata.version == "1.0"
    assert metadata.nonce == "abc123"


def test_key_derivation_dataclass():
    """Test KeyDerivation dataclass structure."""
    kd = KeyDerivation(
        salt="abc123",
        iterations=100000,
        hash_algorithm="SHA256",
        key_length=32,
    )
    assert kd.salt == "abc123"
    assert kd.iterations == 100000
