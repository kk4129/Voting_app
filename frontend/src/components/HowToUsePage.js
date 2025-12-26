import React from 'react';

const HowToUsePage = () => {
    return (
        <div className="page-container">
            <h2>How to Use the Voting Portal</h2>
            
            <div className="card">
                <h3>Step 1: Install MetaMask</h3>
                <p>This platform requires a browser-based crypto wallet. We recommend <a href="https://metamask.io/download/" target="_blank" rel="noopener noreferrer">MetaMask</a>. Install it as a browser extension.</p>
            </div>

            <div className="card">
                <h3>Step 2: Connect to the Network</h3>
                <p>Once installed, your election administrator will provide you with network details (RPC URL, Chain ID) to connect MetaMask to the private election network.</p>
            </div>

            <div className="card">
                <h3>For Election Administrators</h3>
                <ol>
                    <li>Ensure your wallet address is the one that deployed the contract.</li>
                    <li>Navigate to the <strong>Admin Dashboard</strong>.</li>
                    <li>Enter the wallet address of an eligible voter and click "Register Voter".</li>
                    <li>The system will confirm once the voter is successfully registered on the blockchain.</li>
                </ol>
            </div>

            <div className="card">
                <h3>For Voters</h3>
                <ol>
                    <li>Ensure your address has been registered by the administrator.</li>
                    <li>Navigate to the <strong>Voting Booth</strong>.</li>
                    <li>The system will show the list of candidates.</li>
                    <li>Click the "Vote" button for your chosen candidate and confirm the transaction in MetaMask.</li>
                    <li>You can only vote once. The results are updated in real-time.</li>
                </ol>
            </div>
        </div>
    );
};

export default HowToUsePage;
