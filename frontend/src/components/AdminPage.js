import React, { useState, useEffect, useCallback } from 'react';
import { useWeb3 } from '../contexts/Web3Context';
import { ethers } from 'ethers'; // Add this import

function parseList(input) {
  return input.split(',').map(s => s.trim()).filter(Boolean);
}

const AdminPage = () => {
  const { account, isAdmin, contract, owner, getRevertReason, isLoading } = useWeb3();

  // local state
  const [electionName, setElectionName] = useState('');
  const [candidatesCsv, setCandidatesCsv] = useState('');
  const [elections, setElections] = useState([]);
  const [selectedElection, setSelectedElection] = useState(null);
  const [voterAddress, setVoterAddress] = useState('');
  const [bulkVotersCsv, setBulkVotersCsv] = useState('');
  const [message, setMessage] = useState(null);
  const [voterList, setVoterList] = useState([]);
  const [candidates, setCandidates] = useState([]);
  const [transactionLoading, setTransactionLoading] = useState(false);

  const fetchElections = useCallback(async () => {
    if (!contract || isLoading) return;
    
    try {
      console.log('Fetching election count...');
      const count = await contract.electionCount();
      const electionCount = parseInt(count.toString());
      console.log('Total elections:', electionCount);
      
      const arr = [];
      // ✅ FIXED: Start from 1, not 0 (election IDs start at 1)
      for (let i = 1; i <= electionCount; i++) {
        try {
          const name = await contract.getElectionName(i);
          arr.push({ id: i, name });
        } catch (error) {
          console.error(`Error fetching election ${i}:`, error);
          arr.push({ id: i, name: `Election ${i} (Error)` });
        }
      }
      setElections(arr);
      
      if (arr.length > 0 && !selectedElection) {
        setSelectedElection(arr[0].id);
      }
    } catch (error) {
      console.error('Error fetching elections:', error);
      setElections([]);
      setMessage({ type: 'error', text: `Failed to fetch elections: ${error.message}` });
    }
  }, [contract, isLoading, selectedElection]);

  const fetchSelectedDetails = useCallback(async (id) => {
    if (!contract || id === null || id === undefined) return;
    
    try {
      console.log('Fetching details for election ID:', id);
      
      // Fetch candidates
      const cand = await contract.getCandidates(id);
      setCandidates(cand || []);
      
      // Fetch voters (may fail if not admin or function not available)
      try {
        const voters = await contract.getVoters(id);
        setVoterList(voters || []);
      } catch (voterError) {
        console.log('Could not fetch voters (may be normal):', voterError.message);
        setVoterList([]);
      }
    } catch (error) {
      console.error('Error fetching election details:', error);
      setCandidates([]);
      setVoterList([]);
    }
  }, [contract]);

  useEffect(() => {
    if (contract && !isLoading) {
      fetchElections();
    }
  }, [contract, isLoading, fetchElections]);

  useEffect(() => {
    if (selectedElection !== null && contract) {
      fetchSelectedDetails(selectedElection);
    }
  }, [selectedElection, contract, fetchSelectedDetails]);

  const showError = (text) => {
    setMessage({ type: 'error', text });
    setTimeout(() => setMessage(null), 5000);
  };

  const showSuccess = (text) => {
    setMessage({ type: 'success', text });
    setTimeout(() => setMessage(null), 5000);
  };

  const handleCreateElection = async () => {
    if (!contract || !isAdmin) {
      showError('Only admin can create elections.');
      return;
    }

    const names = parseList(candidatesCsv);
    if (!electionName.trim()) {
      showError('Please enter election name.');
      return;
    }
    if (names.length === 0) {
      showError('Please add at least one candidate.');
      return;
    }

    setTransactionLoading(true);
    
    try {
      console.log('Creating election:', { name: electionName, candidates: names });
      showSuccess('Creating election...');

      // Estimate gas first
      let gasEstimate;
      try {
        gasEstimate = await contract.estimateGas.createElection(electionName, names);
        console.log('Gas estimate:', gasEstimate.toString());
      } catch (gasError) {
        console.warn('Gas estimation failed, using default:', gasError.message);
        gasEstimate = ethers.BigNumber.from(500000); // Fallback
      }

      // Send transaction
      const tx = await contract.createElection(electionName, names, {
        gasLimit: gasEstimate.mul(12).div(10) // Add 20% buffer
      });

      showSuccess(`Transaction submitted: ${tx.hash.substring(0, 10)}...`);
      console.log('Transaction sent:', tx.hash);

      // Wait for confirmation
      const receipt = await tx.wait();
      console.log('Transaction confirmed:', receipt);

      showSuccess(`✅ Election "${electionName}" created successfully!`);
      
      // Reset form
      setElectionName('');
      setCandidatesCsv('');
      
      // Refresh data
      await fetchElections();
      
    } catch (err) {
      console.error('Create election error:', err);
      const errorMsg = getRevertReason ? getRevertReason(err) : err.message;
      showError(`Failed to create election: ${errorMsg}`);
    } finally {
      setTransactionLoading(false);
    }
  };

  const handleRegisterVoter = async () => {
    if (!contract || !isAdmin || selectedElection === null) {
      showError('Select election and ensure you are admin.');
      return;
    }
    
    if (!voterAddress.trim()) {
      showError('Enter voter address.');
      return;
    }

    setTransactionLoading(true);
    
    try {
      showSuccess('Registering voter...');
      const tx = await contract.registerVoterForElection(selectedElection, voterAddress);
      
      showSuccess(`Transaction submitted: ${tx.hash.substring(0, 10)}...`);
      await tx.wait();
      
      showSuccess('✅ Voter registered successfully!');
      setVoterAddress('');
      await fetchSelectedDetails(selectedElection);
      
    } catch (err) {
      console.error('Register voter error:', err);
      const errorMsg = getRevertReason ? getRevertReason(err) : err.message;
      showError(`Failed to register voter: ${errorMsg}`);
    } finally {
      setTransactionLoading(false);
    }
  };

  const handleRegisterBulk = async () => {
    if (!contract || !isAdmin || selectedElection === null) {
      showError('Select election and ensure you are admin.');
      return;
    }
    
    const list = parseList(bulkVotersCsv);
    if (list.length === 0) {
      showError('Enter addresses separated by commas.');
      return;
    }

    setTransactionLoading(true);
    
    try {
      showSuccess('Registering voters...');
      const tx = await contract.registerVotersForElection(selectedElection, list);
      
      showSuccess(`Transaction submitted: ${tx.hash.substring(0, 10)}...`);
      await tx.wait();
      
      showSuccess(`✅ ${list.length} voters registered successfully!`);
      setBulkVotersCsv('');
      await fetchSelectedDetails(selectedElection);
      
    } catch (err) {
      console.error('Bulk register error:', err);
      const errorMsg = getRevertReason ? getRevertReason(err) : err.message;
      showError(`Failed to register voters: ${errorMsg}`);
    } finally {
      setTransactionLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="page-container">
        <h2>Admin — Manage Elections</h2>
        <div className="card" style={{ padding: 16 }}>
          <p>Loading blockchain data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container">
      <h2>Admin — Manage Elections</h2>

      {!account && (
        <div className="card" style={{ padding: 16 }}>
          <p>Please connect your wallet.</p>
        </div>
      )}

      {account && !isAdmin && (
        <div className="card" style={{ 
          borderLeft: '4px solid #dc2626', 
          backgroundColor: '#fff5f5', 
          color: '#7f1d1d', 
          padding: 16 
        }}>
          <h3 style={{ margin: 0 }}>Access Denied</h3>
          <p style={{ margin: '8px 0 0' }}>
            Connect with the admin account to access this page.
          </p>
          <p style={{ margin: '8px 0 0', fontSize: 12 }}>
            Contract owner: <strong style={{ color: '#7f1d1d' }}>{owner || 'not loaded'}</strong>
          </p>
          <p style={{ margin: '8px 0 0', fontSize: 12 }}>
            Your account: <strong style={{ color: '#7f1d1d' }}>{account}</strong>
          </p>
        </div>
      )}

      {account && isAdmin && (
        <>
          <div className="card" style={{ padding: 16, marginBottom: 16 }}>
            <h3>Create New Election</h3>
            <input 
              type="text" 
              placeholder="Election name" 
              value={electionName} 
              onChange={e => setElectionName(e.target.value)}
              style={{ marginBottom: 8, width: '100%', padding: 8 }}
              disabled={transactionLoading}
            />
            <input 
              type="text" 
              placeholder="Candidates (comma separated, e.g., Alice, Bob, Charlie)" 
              value={candidatesCsv} 
              onChange={e => setCandidatesCsv(e.target.value)}
              style={{ marginBottom: 8, width: '100%', padding: 8 }}
              disabled={transactionLoading}
            />
            <button 
              onClick={handleCreateElection}
              disabled={transactionLoading || !electionName.trim() || parseList(candidatesCsv).length === 0}
              style={{ padding: '8px 16px' }}
            >
              {transactionLoading ? 'Creating...' : 'Create Election'}
            </button>
            <p style={{ fontSize: 12, color: '#666', marginTop: 8 }}>
              Note: Election IDs start from 1. After creation, refresh will show new election.
            </p>
          </div>

          <div className="card" style={{ padding: 16 }}>
            <h3>Existing Elections</h3>
            <select 
              value={selectedElection || ''} 
              onChange={e => setSelectedElection(parseInt(e.target.value))}
              style={{ width: '100%', padding: 8, marginBottom: 16 }}
              disabled={transactionLoading || elections.length === 0}
            >
              {elections.length === 0 ? (
                <option value="">No elections available</option>
              ) : (
                <>
                  <option value="">Select an election</option>
                  {elections.map(ev => (
                    <option key={ev.id} value={ev.id}>
                      #{ev.id} — {ev.name}
                    </option>
                  ))}
                </>
              )}
            </select>
            
            {selectedElection && elections.find(e => e.id === selectedElection) && (
              <>
                <div style={{ marginBottom: 16 }}>
                  <h4>Candidates ({candidates.length})</h4>
                  {candidates.length === 0 ? (
                    <p style={{ color: '#666' }}>No candidates in this election</p>
                  ) : (
                    <ul style={{ paddingLeft: 20 }}>
                      {candidates.map((c, i) => (
                        <li key={i}>
                          <strong>#{i}:</strong> {c}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>

                <div style={{ marginBottom: 16 }}>
                  <h4>Register Voters</h4>
                  <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
                    <input 
                      type="text" 
                      placeholder="Single voter address (0x...)" 
                      value={voterAddress} 
                      onChange={e => setVoterAddress(e.target.value)}
                      style={{ flex: 1, padding: 8 }}
                      disabled={transactionLoading}
                    />
                    <button 
                      onClick={handleRegisterVoter}
                      disabled={transactionLoading || !voterAddress.trim()}
                      style={{ padding: '8px 16px' }}
                    >
                      {transactionLoading ? 'Processing...' : 'Register'}
                    </button>
                  </div>
                  
                  <p style={{ margin: '12px 0 8px', fontSize: 14 }}>Or bulk register (comma separated):</p>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <input 
                      type="text" 
                      placeholder="addr1, addr2, addr3, ..." 
                      value={bulkVotersCsv} 
                      onChange={e => setBulkVotersCsv(e.target.value)}
                      style={{ flex: 1, padding: 8 }}
                      disabled={transactionLoading}
                    />
                    <button 
                      onClick={handleRegisterBulk}
                      disabled={transactionLoading || parseList(bulkVotersCsv).length === 0}
                      style={{ padding: '8px 16px' }}
                    >
                      {transactionLoading ? 'Processing...' : 'Register Bulk'}
                    </button>
                  </div>
                </div>

                <div>
                  <h4>Registered Voters ({voterList.length})</h4>
                  {voterList.length === 0 ? (
                    <p style={{ color: '#666' }}>No voters registered yet</p>
                  ) : (
                    <ul className="voter-list" style={{ 
                      maxHeight: 200, 
                      overflowY: 'auto', 
                      padding: 8, 
                      backgroundColor: '#f9f9f9',
                      borderRadius: 4 
                    }}>
                      {voterList.map((v, i) => (
                        <li key={i} style={{ 
                          padding: '4px 0', 
                          borderBottom: i < voterList.length - 1 ? '1px solid #eee' : 'none',
                          fontFamily: 'monospace',
                          fontSize: 12
                        }}>
                          {v}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </>
            )}
          </div>
        </>
      )}

      {message && (
        <div
          className="card"
          style={{
            marginTop: 16,
            padding: 12,
            borderLeft: message.type === 'error' ? '4px solid #dc2626' : '4px solid #16a34a',
            backgroundColor: message.type === 'error' ? '#fff5f5' : '#f0fff4',
            color: message.type === 'error' ? '#7f1d1d' : '#065f46'
          }}
        >
          <p style={{ margin: 0, fontWeight: message.type === 'error' ? 'bold' : 'normal' }}>
            {message.text}
          </p>
        </div>
      )}

      {/* Debug info */}
      <div style={{ marginTop: 20, fontSize: 12, color: '#666' }}>
        <p>Debug: Contract {contract ? '✓ Loaded' : '✗ Not loaded'} | 
           Admin: {isAdmin ? '✓ Yes' : '✗ No'} | 
           Account: {account ? `${account.substring(0, 10)}...` : 'Not connected'} |
           Elections: {elections.length}
        </p>
      </div>
    </div>
  );
};

export default AdminPage;