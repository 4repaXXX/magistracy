# Submission stage:

# 1. Alice signs the message with her digital signature and encrypts it with Bob's public key (an asymmetric algorithm).
# 2. Alice generates a random session key and encrypts the message with this key (using a symmetric algorithm).
# 3. The session key is encrypted with Bob's public key (asymmetric algorithm).
# Alice sends Bob an encrypted message, a signature, and an encrypted session key.

# Reception stage:

# Bob receives Alice's encrypted message, signature, and encrypted session key.
# 4. Bob decrypts the session key with his private key.
# 5. Using the thus obtained session key, Bob decrypts Alice’s encrypted message.
# 6. Bob decrypts and verifies Alice's signature.

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature
import os


def alice_process(message, bob_public_key, alice_private_key):
    cipher_rsa = bob_public_key.encrypt(
        message,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    signer = alice_private_key.signer(
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256(),
    )
    signer.update(message)
    signature = signer.finalize()

    session_key = os.urandom(16)
    iv = os.urandom(16)
    cipher_aes = Cipher(
        algorithms.AES(session_key), modes.CFB(iv), backend=default_backend()
    )
    encryptor = cipher_aes.encryptor()
    ciphertext = iv + encryptor.update(message) + encryptor.finalize()

    # Encrypt the session key with Bob's public key
    encrypted_session_key = bob_public_key.encrypt(
        session_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    return cipher_rsa, signature, encrypted_session_key, ciphertext


def bob_process(encrypted_message, signature, encrypted_session_key, ciphertext, bob_private_key, alice_public_key):
    decrypted_session_key = bob_private_key.decrypt(
        encrypted_session_key,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
    )

    iv = ciphertext[:16]
    cipher_aes = Cipher(algorithms.AES(decrypted_session_key), modes.CFB(iv), backend=default_backend())
    decryptor = cipher_aes.decryptor()
    decrypted_message = decryptor.update(ciphertext[16:]) + decryptor.finalize()

    verifier = alice_public_key.verifier(signature, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
    verifier.update(decrypted_message)
    try:
        verifier.verify()
        print("The signature is valid.")
    except InvalidSignature:
        print("The signature is not valid.")

    return decrypted_message


def main():
    alice_private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )
    bob_private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )
    bob_public_key = bob_private_key.public_key()
    alice_public_key = alice_private_key.public_key()

    input_path = input("Enter the path of the file to encrypt: ")
    output_path = input("Enter the path for the output file: ")

    if input_path:
        with open(input_path, 'rb') as file:
            message = file.read()
    else:
        message = b"Hello, this is a test message."

    encrypted_message, signature, encrypted_session_key, ciphertext = alice_process(message, bob_public_key, alice_private_key)
    decrypted_message = bob_process(encrypted_message, signature, encrypted_session_key, ciphertext, bob_private_key, alice_public_key)

    if output_path:
        with open(output_path, 'wb') as file:
            file.write(decrypted_message)
        print(f"Decrypted message written to {output_path}")
    else:
        print("Original Message:", message)
        print("Decrypted Message:", decrypted_message)


if __name__ == "__main__":
    main()
