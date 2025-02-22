// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract VFLPrivacyManagement {
    // Mapping to store consent status of participants
    mapping(address => bool) public consentStatus;

    // Event to log consent
    event ConsentGiven(address participant);

    // Function to give consent
    function giveConsent() public {
        require(!consentStatus[msg.sender], "Consent already given.");
        consentStatus[msg.sender] = true;
        emit ConsentGiven(msg.sender);
    }

    // Function to check if a participant has given consent
    function hasConsent(address participant) public view returns (bool) {
        return consentStatus[participant];
    }
}