import React, { useState, useEffect, useCallback } from 'react';
import { useWeb3 } from '../contexts/Web3Context';
import { ethers } from 'ethers';

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
  const [voteCounts, setVoteCounts] = useState({});
  const [totalVotes, setTotalVotes] = useState(0);
  const [transactionLoading, setTransactionLoading] = useState(false);

  const fetchElections = useCallback(async () => {
    if (!contract || isLoading) return;
    
    try {
      console.log('Fetching election count...');
      const count = await contract.electionCount();
      const electionCount = parseInt(count.toString());
      console.log('Total elections:', electionCount);
      
      const arr = [];
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
      
      const cand = await contract.getCandidates(id);
      setCandidates(cand || []);
      
      const counts = {};
      let total = 0;
      
      if (cand && cand.length > 0) {
        for (let i = 0; i < cand.length; i++) {
          try {
            const votes = await contract.getVotes(id, i);
            counts[i] = parseInt(votes.toString());
            total += counts[i];
          } catch (error) {
            console.error(`Error fetching votes for candidate ${i}:`, error);
            counts[i] = 0;
          }
        }
      }
      
      setVoteCounts(counts);
      setTotalVotes(total);
      
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
      setVoteCounts({});
      setTotalVotes(0);
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

  useEffect(() => {
    if (!selectedElection || !contract) return;
    
    const interval = setInterval(() => {
      fetchSelectedDetails(selectedElection);
    }, 10000);
    
    return () => clearInterval(interval);
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

      let gasEstimate;
      try {
        gasEstimate = await contract.estimateGas.createElection(electionName, names);
        console.log('Gas estimate:', gasEstimate.toString());
      } catch (gasError) {
        console.warn('Gas estimation failed, using default:', gasError.message);
        gasEstimate = ethers.BigNumber.from(500000);
      }

      const tx = await contract.createElection(electionName, names, {
        gasLimit: gasEstimate.mul(12).div(10)
      });

      showSuccess(`Transaction submitted: ${tx.hash.substring(0, 10)}...`);
      console.log('Transaction sent:', tx.hash);

      const receipt = await tx.wait();
      console.log('Transaction confirmed:', receipt);

      showSuccess(`✅ Election "${electionName}" created successfully!`);
      
      setElectionName('');
      setCandidatesCsv('');
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

    // Validate and checksum the address to avoid ENS resolution
    let checksummedAddress;
    try {
      checksummedAddress = ethers.utils.getAddress(voterAddress.trim());
    } catch (e) {
      showError('Invalid Ethereum address format.');
      return;
    }

    setTransactionLoading(true);
    
    try {
      showSuccess('Registering voter...');
      const tx = await contract.registerVoterForElection(selectedElection, checksummedAddress);
      
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

    // Validate all addresses first to avoid ENS resolution
    let checksummedList;
    try {
      checksummedList = list.map(addr => ethers.utils.getAddress(addr.trim()));
    } catch (e) {
      showError('One or more addresses are invalid.');
      return;
    }

    setTransactionLoading(true);
    
    try {
      showSuccess('Registering voters...');
      const tx = await contract.registerVotersForElection(selectedElection, checksummedList);
      
      showSuccess(`Transaction submitted: ${tx.hash.substring(0, 10)}...`);
      await tx.wait();
      
      showSuccess(`✅ ${checksummedList.length} voters registered successfully!`);
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

  const refreshVoteCounts = () => {
    if (selectedElection) {
      fetchSelectedDetails(selectedElection);
      showSuccess('Vote counts refreshed!');
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
          </div>

          <div className="card" style={{ padding: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <h3 style={{ margin: 0 }}>Election Dashboard</h3>
              <button 
                onClick={refreshVoteCounts}
                style={{ 
                  padding: '6px 12px', 
                  backgroundColor: '#10b981',
                  color: 'white',
                  border: 'none',
                  borderRadius: 4,
                  cursor: 'pointer',
                  fontSize: 14
                }}
              >
                🔄 Refresh Votes
              </button>
            </div>
            
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
                <div style={{ 
                  backgroundColor: '#f0f9ff', 
                  border: '1px solid #bae6fd',
                  borderRadius: 8,
                  padding: 16,
                  marginBottom: 20
                }}>
                  <h4 style={{ marginTop: 0, color: '#0369a1' }}>📊 Voting Statistics</h4>
                  <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 24, fontWeight: 'bold', color: '#0369a1' }}>{totalVotes}</div>
                      <div style={{ fontSize: 12, color: '#64748b' }}>Total Votes</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 24, fontWeight: 'bold', color: '#0369a1' }}>{voterList.length}</div>
                      <div style={{ fontSize: 12, color: '#64748b' }}>Registered Voters</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 24, fontWeight: 'bold', color: '#0369a1' }}>
                        {voterList.length > 0 ? `${((totalVotes / voterList.length) * 100).toFixed(1)}%` : '0%'}
                      </div>
                      <div style={{ fontSize: 12, color: '#64748b' }}>Voter Turnout</div>
                    </div>
                  </div>
                </div>

                <div style={{ marginBottom: 20 }}>
                  <h4>Candidates ({candidates.length})</h4>
                  {candidates.length === 0 ? (
                    <p style={{ color: '#666' }}>No candidates in this election</p>
                  ) : (
                    <div style={{ display: 'grid', gap: 12 }}>
                      {candidates.map((candidate, index) => (
                        <div key={index} style={{ 
                          border: '1px solid #e2e8f0',
                          borderRadius: 8,
                          padding: 16,
                          backgroundColor: '#f8fafc'
                        }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <div>
                              <strong style={{ fontSize: 16 }}>#{index}: {candidate}</strong>
                              <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
                                Candidate ID: {index}
                              </div>
                            </div>
                            <div style={{ textAlign: 'right' }}>
                              <div style={{ fontSize: 24, fontWeight: 'bold', color: '#059669' }}>
                                {voteCounts[index] || 0}
                              </div>
                              <div style={{ fontSize: 12, color: '#64748b' }}>Votes</div>
                            </div>
                          </div>
                          
                          {totalVotes > 0 && (
                            <div style={{ marginTop: 12 }}>
                              <div style={{ 
                                width: '100%', 
                                height: 8, 
                                backgroundColor: '#e2e8f0',
                                borderRadius: 4,
                                overflow: 'hidden'
                              }}>
                                <div 
                                  style={{ 
                                    width: `${((voteCounts[index] || 0) / totalVotes) * 100}%`,
                                    height: '100%',
                                    backgroundColor: '#10b981',
                                    borderRadius: 4
                                  }}
                                />
                              </div>
                              <div style={{ 
                                fontSize: 12, 
                                color: '#64748b',
                                marginTop: 4,
                                textAlign: 'right'
                              }}>
                                {totalVotes > 0 ? `${(((voteCounts[index] || 0) / totalVotes) * 100).toFixed(1)}%` : '0%'} of total votes
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div style={{ marginBottom: 20, padding: 16, backgroundColor: '#fefce8', border: '1px solid #fde047', borderRadius: 8 }}>
                  <h4 style={{ color: '#ca8a04', marginTop: 0 }}>Register Voters</h4>
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
                      style={{ 
                        padding: '8px 16px',
                        backgroundColor: '#f59e0b',
                        color: 'white',
                        border: 'none',
                        borderRadius: 4
                      }}
                    >
                      {transactionLoading ? 'Processing...' : 'Register'}
                    </button>
                  </div>
                  
                  <p style={{ margin: '12px 0 8px', fontSize: 14, color: '#92400e' }}>Or bulk register (comma separated):</p>
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
                      style={{ 
                        padding: '8px 16px',
                        backgroundColor: '#f59e0b',
                        color: 'white',
                        border: 'none',
                        borderRadius: 4
                      }}
                    >
                      {transactionLoading ? 'Processing...' : 'Register Bulk'}
                    </button>
                  </div>
                </div>

                <div style={{ marginTop: 20 }}>
                  <h4>Registered Voters ({voterList.length})</h4>
                  {voterList.length === 0 ? (
                    <p style={{ color: '#666' }}>No voters registered yet</p>
                  ) : (
                    <div style={{ 
                      maxHeight: 200, 
                      overflowY: 'auto', 
                      padding: 8, 
                      backgroundColor: '#f9f9f9',
                      borderRadius: 4 
                    }}>
                      {voterList.map((v, i) => (
                        <div key={i} style={{ 
                          padding: '8px 0', 
                          borderBottom: i < voterList.length - 1 ? '1px solid #eee' : 'none',
                          fontFamily: 'monospace',
                          fontSize: 12,
                          display: 'flex',
                          alignItems: 'center',
                          gap: 8
                        }}>
                          <div style={{ 
                            width: 20, 
                            height: 20, 
                            backgroundColor: '#3b82f6',
                            color: 'white',
                            borderRadius: '50%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: 10
                          }}>
                            {i + 1}
                          </div>
                          {v}
                        </div>
                      ))}
                    </div>
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

      <div style={{ marginTop: 20, fontSize: 12, color: '#666', padding: 12, backgroundColor: '#f8fafc', borderRadius: 6 }}>
        <p style={{ margin: 0 }}>
          <strong>Status:</strong> Contract {contract ? '✓' : '✗'} | 
          Admin: {isAdmin ? '✓' : '✗'} | 
          Account: {account ? `${account.substring(0, 10)}...` : 'Not connected'} |
          Elections: {elections.length} | 
          Selected: {selectedElection || 'None'}
        </p>
      </div>
    </div>
  );
};

export default AdminPage;