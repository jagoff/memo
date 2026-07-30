import CryptoKit
import Foundation
import Security

private enum HelperError: Error {
    case message(String)
}

private func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data(message.utf8))
    exit(1)
}

private let serviceRoot = "com.memo.operational-signing"
private let keychainMarker = Data("memo.secure-enclave.p256.v1".utf8)

private func isASCIIAlphaNumeric(_ scalar: Unicode.Scalar) -> Bool {
    (scalar.value >= 48 && scalar.value <= 57)
        || (scalar.value >= 65 && scalar.value <= 90)
        || (scalar.value >= 97 && scalar.value <= 122)
}

private func validService(_ service: String) -> Bool {
    if service == serviceRoot {
        return true
    }
    guard service.hasPrefix(serviceRoot + ".") else {
        return false
    }
    let suffix = service.dropFirst(serviceRoot.count + 1)
    let segments = suffix.split(separator: ".", omittingEmptySubsequences: false)
    guard !segments.isEmpty && segments.count <= 4 else {
        return false
    }
    return segments.allSatisfy { segment in
        let scalars = Array(segment.unicodeScalars)
        return !scalars.isEmpty
            && scalars.count <= 63
            && isASCIIAlphaNumeric(scalars[0])
            && scalars.allSatisfy {
                isASCIIAlphaNumeric($0) || $0.value == 45
            }
    }
}

private func validKeyID(_ keyID: String) -> Bool {
    let prefix = "p256-se-"
    guard keyID.hasPrefix(prefix) else {
        return false
    }
    let suffix = keyID.dropFirst(prefix.count)
    return suffix.utf8.count == 32 && suffix.utf8.allSatisfy {
        ($0 >= 48 && $0 <= 57) || ($0 >= 97 && $0 <= 102)
    }
}

private func query(service: String, keyID: String) -> [String: Any] {
    [
        kSecClass as String: kSecClassGenericPassword,
        kSecAttrService as String: service,
        kSecAttrAccount as String: keyID,
        kSecAttrGeneric as String: keychainMarker,
        kSecAttrSynchronizable as String: false,
    ]
}

private func readWrappedKey(service: String, keyID: String) throws -> Data {
    var request = query(service: service, keyID: keyID)
    request[kSecReturnData as String] = true
    request[kSecMatchLimit as String] = kSecMatchLimitOne
    var value: CFTypeRef?
    let status = SecItemCopyMatching(request as CFDictionary, &value)
    if status == errSecItemNotFound {
        throw HelperError.message("unknown private key id")
    }
    guard status == errSecSuccess, let data = value as? Data else {
        throw HelperError.message("Keychain read failed")
    }
    return data
}

private func generate(service: String, keyID: String) throws {
    do {
        _ = try readWrappedKey(service: service, keyID: keyID)
        throw HelperError.message("duplicate private key id")
    } catch HelperError.message(let message) where message == "unknown private key id" {
        // Expected absence.
    }
    guard SecureEnclave.isAvailable else {
        throw HelperError.message("Secure Enclave is unavailable")
    }
    let key: SecureEnclave.P256.Signing.PrivateKey
    do {
        key = try SecureEnclave.P256.Signing.PrivateKey()
    } catch {
        throw HelperError.message("Secure Enclave key generation failed")
    }
    var request = query(service: service, keyID: keyID)
    request[kSecValueData as String] = key.dataRepresentation
    request[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
    let status = SecItemAdd(request as CFDictionary, nil)
    if status == errSecDuplicateItem {
        throw HelperError.message("duplicate private key id")
    }
    guard status == errSecSuccess else {
        throw HelperError.message("Keychain write failed")
    }
    FileHandle.standardOutput.write(key.publicKey.x963Representation)
}

private func sign(service: String, keyID: String) throws {
    let wrapped = try readWrappedKey(service: service, keyID: keyID)
    let key: SecureEnclave.P256.Signing.PrivateKey
    do {
        key = try SecureEnclave.P256.Signing.PrivateKey(dataRepresentation: wrapped)
    } catch {
        throw HelperError.message("Secure Enclave key recovery failed")
    }
    let payload = FileHandle.standardInput.readDataToEndOfFile()
    do {
        let signature = try key.signature(for: payload)
        FileHandle.standardOutput.write(signature.derRepresentation)
    } catch {
        throw HelperError.message("Secure Enclave signing failed")
    }
}

private func destroy(service: String, keyID: String) throws {
    let status = SecItemDelete(query(service: service, keyID: keyID) as CFDictionary)
    if status == errSecItemNotFound {
        throw HelperError.message("unknown private key id")
    }
    guard status == errSecSuccess else {
        throw HelperError.message("Keychain delete failed")
    }
}

guard CommandLine.arguments.count == 4 else {
    fail("Secure Enclave helper operation failed")
}
let operation = CommandLine.arguments[1]
let service = CommandLine.arguments[2]
let keyID = CommandLine.arguments[3]
guard validService(service) else {
    fail("invalid Keychain service")
}
guard validKeyID(keyID) else {
    fail("invalid private key id")
}

do {
    switch operation {
    case "generate":
        try generate(service: service, keyID: keyID)
    case "sign":
        try sign(service: service, keyID: keyID)
    case "destroy":
        try destroy(service: service, keyID: keyID)
    default:
        throw HelperError.message("Secure Enclave helper operation failed")
    }
} catch HelperError.message(let message) {
    fail(message)
} catch {
    fail("Secure Enclave helper operation failed")
}
