from web3 import Web3
import json

# Load ABI from file
with open('abi1.json', 'r') as f:
    abi = json.load(f)

# Connect to Ganache (local Ethereum blockchain)
ganache_url = "HTTP://127.0.0.1:7545"  
web3 = Web3(Web3.HTTPProvider(ganache_url))

# Check if connected
if not web3.is_connected():
    raise Exception("Failed to connect to Ganache")


# Set default account (first account in Ganache)
web3.eth.defaultAccount = web3.eth.accounts[0]

# Load the contract
contract_address = web3.to_checksum_address("0x830a5108950162180fd57561486db3e2f9751824")
contract = web3.eth.contract(address=contract_address, abi=abi)

def give_consent(participant_address):
    """
    Give consent on behalf of a participant.
    """
    try:
        # Check if consent is already given
        if has_consent(participant_address):
            print(f"Consent already given by: {participant_address}")
            return

        # Send transaction to give consent
        tx_hash = contract.functions.giveConsent().transact({'from': participant_address})
        tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"Consent given by: {participant_address}")
        print(f"Transaction receipt: {tx_receipt}")
    except Exception as e:
        print(f"Error giving consent: {e}")



def has_consent(participant_address):
    """
    Check if a participant has given consent.
    """
    try:
        # Call the smart contract to check consent
        consent_status = contract.functions.hasConsent(participant_address).call()
        print(f"Consent status for {participant_address}: {consent_status}")
        return consent_status
    except Exception as e:
        print(f"Error checking consent: {e}")
        return False

if __name__ == "__main__":
    # Assign Ethereum addresses to participants
    party_a_address = web3.to_checksum_address(web3.eth.accounts[1])
    party_b_address = web3.to_checksum_address(web3.eth.accounts[2])
  

    # Give consent
    give_consent(party_a_address)
    give_consent(party_b_address)

    # Check consent
    print("\nChecking consent status:")
    has_consent(party_a_address)
    has_consent(party_b_address)