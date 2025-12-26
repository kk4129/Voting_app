import React, { useState, useEffect } from 'react';
import { useWeb3 } from '../contexts/Web3Context';

const VotingPage = () => {
  const { contract, account, isLoading, error, getRevertReason } = useWeb3();
  const [elections, setElections] = useState([]);
  const [loading, setLoading] = useState(true);
  const [pageError, setPageError] = useState(null);
  const [votingInProgress, setVotingInProgress] = useState(false);

  useEffect(() => {
    const fetchElections = async () => {
      // Don't try to fetch if contract isn't loaded yet
      if (!contract || isLoading) {
        console.log('Waiting for contract to load...');
        return;
      }

      setLoading(true);
      setPageError(null);
      
      try {
        console.log('Fetching elections, contract available:', !!contract);
        
        // First get the election count
        const count = await contract.electionCount();
        const totalElections = parseInt(count.toString());
        console.log('Election count:', totalElections);
        
        const electionArray = [];
        
        // ✅ FIXED: Start from 1, not 0 (election IDs start at 1)
        for (let i = 1; i <= totalElections; i++) {
          try {
            // Get election name
            const name = await contract.getElectionName(i);
            
            // Get candidates for this election
            const candidates = await contract.getCandidates(i);
            
            electionArray.push({
              id: i, // ✅ Store actual election ID (1, 2, 3...)
              name: name,
              candidates: candidates,
              candidateCount: candidates.length
            });
          } catch (electionError) {
            console.error(`Error fetching election ${i}:`, electionError);
            // Continue with next election
          }
        }
        
        setElections(electionArray);
        console.log('Elections fetched:', electionArray);
        
      } catch (error) {
        console.error('Error fetching elections:', error);
        setPageError(error.message || 'Failed to fetch elections');
      } finally {
        setLoading(false);
      }
    };

    fetchElections();
  }, [contract, isLoading]);

  const handleVote = async (electionId, candidateIndex, candidateName, electionName) => {
    if (!contract || !account) {
      alert('Please connect your wallet first');
      return;
    }

    if (votingInProgress) {
      alert('Please wait for previous vote to complete');
      return;
    }

    const confirmVote = window.confirm(`Are you sure you want to vote for "${candidateName}" in "${electionName}"?`);
    if (!confirmVote) return;

    setVotingInProgress(true);
    
    try {
      console.log(`Voting: Election ${electionId}, Candidate ${candidateIndex}`);
      
      // Estimate gas first
      let gasEstimate;
      try {
        gasEstimate = await contract.estimateGas.vote(electionId, candidateIndex);
        console.log('Gas estimate:', gasEstimate.toString());
      } catch (gasError) {
        console.warn('Gas estimation failed:', gasError.message);
      }

      // Send vote transaction
      const tx = await contract.vote(electionId, candidateIndex, {
        gasLimit: gasEstimate ? gasEstimate.mul(12).div(10) : undefined // 20% buffer
      });

      console.log('Voting transaction sent:', tx.hash);
      alert(`Vote submitted! Transaction: ${tx.hash.substring(0, 10)}...\nWaiting for confirmation...`);
      
      // Wait for confirmation
      const receipt = await tx.wait();
      console.log('Vote confirmed:', receipt);
      
      alert(`✅ Successfully voted for "${candidateName}" in "${electionName}"!`);
      
      // Refresh elections to show updated state
      setLoading(true);
      setTimeout(() => {
        window.location.reload(); // Simple refresh to update data
      }, 2000);
      
    } catch (err) {
      console.error('Vote failed:', err);
      
      // Get user-friendly error message
      let errorMsg = err.message;
      if (getRevertReason) {
        errorMsg = getRevertReason(err);
      }
      
      // Common vote errors
      if (errorMsg.includes('already voted')) {
        alert(`You have already voted in "${electionName}"`);
      } else if (errorMsg.includes('not registered')) {
        alert(`You are not registered to vote in "${electionName}"`);
      } else if (errorMsg.includes('rejected')) {
        alert('Transaction was rejected in MetaMask');
      } else {
        alert(`Vote failed: ${errorMsg}`);
      }
    } finally {
      setVotingInProgress(false);
    }
  };

  // Show loading state
  if (isLoading) {
    return (
      <div className="voting-page" style={{ padding: 20 }}>
        <h2>Loading voting data...</h2>
        <p>Connecting to blockchain...</p>
      </div>
    );
  }

  // Show error state
  if (error || pageError) {
    return (
      <div className="voting-page" style={{ padding: 20 }}>
        <h2>Error</h2>
        <p style={{ color: 'red' }}>{error || pageError}</p>
        <p>Please check your MetaMask connection and try again.</p>
      </div>
    );
  }

  // Show no account connected
  if (!account) {
    return (
      <div className="voting-page" style={{ padding: 20 }}>
        <h2>Connect Wallet</h2>
        <p>Please connect your wallet to view and participate in elections.</p>
      </div>
    );
  }

  return (
    <div className="voting-page" style={{ padding: 20 }}>
      <h2>Available Elections</h2>
      
      {loading ? (
        <p>Loading elections...</p>
      ) : elections.length === 0 ? (
        <div style={{ padding: 20, textAlign: 'center', backgroundColor: '#f9f9f9', borderRadius: 8 }}>
          <p style={{ fontSize: 18, marginBottom: 10 }}>No elections available yet.</p>
          <p>As an admin, you can create elections from the Admin page.</p>
          <p>As a voter, please wait for elections to be created.</p>
        </div>
      ) : (
        <div className="elections-list" style={{ display: 'grid', gap: 20 }}>
          {elections.map((election) => (
            <div key={election.id} className="election-card" style={{ 
              border: '1px solid #ddd', 
              borderRadius: 8, 
              padding: 20,
              backgroundColor: '#fff'
            }}>
              <h3 style={{ marginTop: 0, color: '#333' }}>{election.name}</h3>
              <p style={{ color: '#666' }}>Election ID: {election.id}</p>
              <p style={{ color: '#666' }}>Candidates: {election.candidateCount}</p>
              
              <div className="candidates-list" style={{ marginTop: 15 }}>
                <h4 style={{ marginBottom: 10 }}>Candidates:</h4>
                <ul style={{ listStyle: 'none', padding: 0 }}>
                  {election.candidates.map((candidate, index) => (
                    <li key={index} style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      padding: '10px 0',
                      borderBottom: index < election.candidates.length - 1 ? '1px solid #eee' : 'none'
                    }}>
                      <span style={{ fontSize: 16 }}>{candidate}</span>
                      <button 
                        onClick={() => handleVote(election.id, index, candidate, election.name)}
                        disabled={votingInProgress || loading}
                        style={{
                          padding: '8px 16px',
                          backgroundColor: '#4f46e5',
                          color: 'white',
                          border: 'none',
                          borderRadius: 4,
                          cursor: votingInProgress ? 'not-allowed' : 'pointer',
                          opacity: votingInProgress ? 0.7 : 1
                        }}
                      >
                        {votingInProgress ? 'Processing...' : 'Vote'}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
              
              <div style={{ marginTop: 15, fontSize: 12, color: '#666' }}>
                <p>Connected as: {account.substring(0, 10)}...</p>
              </div>
            </div>
          ))}
        </div>
      )}
      
      {/* Debug info */}
      <div style={{ marginTop: 20, fontSize: 12, color: '#666' }}>
        <p>Debug: Elections loaded: {elections.length} | 
           Account: {account ? 'Connected' : 'Not connected'} |
           Contract: {contract ? '✓' : '✗'}
        </p>
      </div>
    </div>
  );
};

export default VotingPage;