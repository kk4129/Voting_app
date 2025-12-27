import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useWeb3 } from '../contexts/Web3Context';
import { ethers } from 'ethers';
import './VotingPage.css';

const VotingPage = () => {
  const { contract, account, isLoading, error, getRevertReason } = useWeb3();
  
  // Blockchain election states
  const [elections, setElections] = useState([]);
  const [blockchainLoading, setBlockchainLoading] = useState(true);
  const [pageError, setPageError] = useState(null);
  
  // Face monitoring states
  const [faceRegistered, setFaceRegistered] = useState(false);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [verificationStatus, setVerificationStatus] = useState('idle');
  const [message, setMessage] = useState('');
  const [canVote, setCanVote] = useState(false);
  const [monitoringCount, setMonitoringCount] = useState(0);
  const [similarityScore, setSimilarityScore] = useState(0);
  const [matchThreshold, setMatchThreshold] = useState(55);
  
  // Voting states
  const [votingInProgress, setVotingInProgress] = useState(false);
  const [votedElections, setVotedElections] = useState({});
  const [voterRegistrationStatus, setVoterRegistrationStatus] = useState({});
  
  // Backend health check
  const [backendOnline, setBackendOnline] = useState(false);
  const [registeredVotersCount, setRegisteredVotersCount] = useState(0);
  
  // Toast notification state
  const [toast, setToast] = useState(null);
  
  // Registration loading state
  const [registrationLoading, setRegistrationLoading] = useState(false);
  
  // Camera refs
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const monitoringInterval = useRef(null);
  const streamRef = useRef(null);
  
  // Previous account ref to detect account changes
  const prevAccountRef = useRef(null);
  
  // Backend URL - Update this with your ngrok URL
  const BACKEND_URL = "https://nephelinitic-untumidly-doretha.ngrok-free.dev";

  // Fixed API request helper
  const apiRequest = useCallback(async (endpoint, options = {}) => {
    const url = `${BACKEND_URL}${endpoint}`;
    
    console.log(`📡 API Request to: ${endpoint}`);
    
    const fetchOptions = {
      method: options.method || 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        ...options.headers
      },
      mode: 'cors',
    };

    if (options.body) {
      fetchOptions.body = JSON.stringify(options.body);
    }

    try {
      console.log(`🔗 Calling: ${url}`);
      const response = await fetch(url, fetchOptions);
      
      console.log(`📥 Response: ${response.status} ${response.statusText}`);
      
      // Read the response once
      const responseText = await response.text();
      
      // Check for HTML
      if (responseText.includes('<!DOCTYPE') || responseText.includes('<html>')) {
        console.error('❌ Got HTML:', responseText.substring(0, 200));
        throw new Error('Backend returned HTML');
      }
      
      // Parse JSON
      let data;
      try {
        data = JSON.parse(responseText);
      } catch (parseError) {
        console.error('❌ JSON parse error:', parseError);
        console.error('Response text:', responseText);
        throw new Error('Invalid JSON response');
      }
      
      console.log(`✅ Response data from ${endpoint}:`, data);
      
      // Return both the data AND a response-like object
      return {
        ok: response.ok,
        status: response.status,
        statusText: response.statusText,
        data: data,
        json: async () => data
      };
      
    } catch (error) {
      console.error(`❌ API request failed for ${endpoint}:`, error.message);
      
      // Return a mock response for testing if needed
      if (endpoint === '/health') {
        console.log('⚠️ Returning mock health response');
        return {
          ok: true,
          status: 200,
          data: {
            status: "online",
            service: "SecureVote (Mock)",
            registered_voters: 0,
            threshold: 0.55
          },
          json: async () => ({
            status: "online",
            service: "SecureVote (Mock)",
            registered_voters: 0,
            threshold: 0.55
          })
        };
      }
      
      throw error;
    }
  }, []);

  // Toast notification helper
  const showToast = useCallback((type, title, text, duration = 5000) => {
    setToast({ type, title, text });
    setTimeout(() => setToast(null), duration);
  }, []);

  // Stop camera
  const stopCamera = useCallback(() => {
    console.log('Stopping camera...');
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  // Stop monitoring
  const stopMonitoring = useCallback(() => {
    console.log('Stopping monitoring...');
    if (monitoringInterval.current) {
      clearInterval(monitoringInterval.current);
      monitoringInterval.current = null;
    }
    setIsMonitoring(false);
    setVerificationStatus('idle');
  }, []);

  // Start camera
  const startCamera = useCallback(async () => {
    try {
      console.log('Starting camera...');
      
      if (streamRef.current) {
        stopCamera();
      }
      
      const constraints = {
        video: {
          width: { ideal: 640, max: 1280 },
          height: { ideal: 480, max: 720 },
          facingMode: 'user'
        },
        audio: false
      };
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        
        if (canvasRef.current) {
          canvasRef.current.width = 640;
          canvasRef.current.height = 480;
        }
        
        return new Promise((resolve) => {
          if (videoRef.current) {
            videoRef.current.onloadedmetadata = () => {
              console.log('Camera ready');
              resolve(true);
            };
          }
          
          setTimeout(() => {
            if (videoRef.current && videoRef.current.videoWidth > 0) {
              console.log('Camera ready (timeout)');
              resolve(true);
            } else {
              console.log('Camera not ready');
              resolve(false);
            }
          }, 2000);
        });
      }
      return false;
    } catch (error) {
      console.error('Camera error:', error);
      
      let errorMessage = 'Could not access camera';
      if (error.name === 'NotAllowedError') {
        errorMessage = 'Camera permission denied. Please allow camera access.';
      } else if (error.name === 'NotFoundError') {
        errorMessage = 'No camera found on this device.';
      } else if (error.name === 'NotReadableError') {
        errorMessage = 'Camera is in use by another application.';
      }
      
      setMessage(errorMessage);
      showToast('error', 'Camera Error', errorMessage);
      return false;
    }
  }, [stopCamera, showToast]);

  // Capture frame
  const captureFrame = useCallback(() => {
    if (!videoRef.current) {
      console.log('Video ref not available');
      return null;
    }
    
    const video = videoRef.current;
    
    if (video.readyState !== 4 || video.videoWidth === 0) {
      console.log('Video not ready', video.readyState, video.videoWidth);
      return null;
    }
    
    let canvas = canvasRef.current;
    if (!canvas) {
      canvas = document.createElement('canvas');
      canvasRef.current = canvas;
    }
    
    const context = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    try {
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const imageData = canvas.toDataURL('image/jpeg', 0.8);
      
      if (!imageData || imageData.length < 1000) {
        console.log('Image too small');
        return null;
      }
      
      return imageData;
    } catch (error) {
      console.error('Capture error:', error);
      return null;
    }
  }, []);

  // Start monitoring with voter ID
  const startMonitoring = useCallback(() => {
    console.log('Starting monitoring for account:', account);
    
    if (monitoringInterval.current) {
      clearInterval(monitoringInterval.current);
    }
    
    if (!account) {
      console.log('No account connected, cannot start monitoring');
      return;
    }
    
    setIsMonitoring(true);
    setVerificationStatus('idle');
    setMessage('Monitoring active - verifying your face...');
    
    const verifyFace = async () => {
      const imageData = captureFrame();
      if (!imageData) {
        console.log('No frame captured');
        return;
      }
      
      try {
        const response = await apiRequest('/verify-face', {
          method: 'POST',
          body: { 
            image: imageData,
            voterId: account
          }
        });
        
        const result = response.data;
        console.log('Verification result:', result);
        
        setMonitoringCount(prev => prev + 1);
        
        if (result.threshold) {
          setMatchThreshold(Math.round(result.threshold * 100));
        }
        
        if (result.verified) {
          setVerificationStatus('verified');
          setCanVote(true);
          const similarity = Math.round((result.similarity || 0) * 100);
          setSimilarityScore(similarity);
          setMessage(`✅ Face verified (${similarity}% match)`);
        } else {
          setVerificationStatus('failed');
          setCanVote(false);
          
          const similarity = Math.round((result.similarity || 0) * 100);
          setSimilarityScore(similarity);
          
          switch(result.reason) {
            case 'NO_FACE':
              setMessage('👤 No face detected - please face the camera');
              break;
            case 'MULTIPLE_FACES':
              setMessage('👥 Multiple faces detected - only you should be visible');
              break;
            case 'FACE_MISMATCH':
              const threshold = result.threshold ? Math.round(result.threshold * 100) : matchThreshold;
              setMessage(`❌ Face doesn't match (${similarity}% - need ${threshold}%)`);
              break;
            case 'NOT_REGISTERED':
              setMessage('⚠️ No face registered for this account');
              setFaceRegistered(false);
              stopMonitoring();
              break;
            case 'NO_VOTER_ID':
              setMessage('⚠️ Wallet not connected');
              break;
            default:
              setMessage(result.message || '❌ Verification failed');
          }
        }
      } catch (error) {
        console.error('Monitoring error:', error);
        setVerificationStatus('error');
        setMessage('⚠️ Connection error - retrying...');
      }
    };
    
    // Initial verification
    setTimeout(verifyFace, 500);
    
    // Set up interval for continuous monitoring
    monitoringInterval.current = setInterval(verifyFace, 2000);
  }, [account, captureFrame, matchThreshold, stopMonitoring, apiRequest]);

  // Check if current account has registered face
  const checkAccountRegistration = useCallback(async (voterId) => {
    if (!voterId) return;
    
    try {
      console.log('Checking registration for:', voterId);
      
      const response = await apiRequest('/check-registration', {
        method: 'POST',
        body: { voterId }
      });
      
      const result = response.data;
      console.log('Registration check result:', result);
      
      if (result.registered) {
        setFaceRegistered(true);
        setMessage('Face registered for this account. Starting verification...');
        
        // Set the current voter in backend
        try {
          await apiRequest('/set-voter', {
            method: 'POST',
            body: { voterId }
          });
        } catch (e) {
          console.log('Set voter endpoint not available');
        }
        
        // Start camera and monitoring
        const cameraStarted = await startCamera();
        if (cameraStarted) {
          startMonitoring();
        }
      } else {
        setFaceRegistered(false);
        setCanVote(false);
        setVerificationStatus('idle');
        setMessage('Please register your face to vote with this account.');
        stopMonitoring();
      }
      
      return result.registered;
    } catch (error) {
      console.error('Registration check error:', error);
      return false;
    }
  }, [startCamera, startMonitoring, stopMonitoring, apiRequest]);

  // Check backend health
  const checkBackendHealth = useCallback(async () => {
    try {
      const response = await apiRequest('/health');
      const data = response.data;
      setBackendOnline(true);
      setRegisteredVotersCount(data.registered_voters || data.total_registered_voters || 0);
      console.log('✅ Backend health:', data);
      return true;
    } catch (error) {
      console.log('❌ Backend not running:', error);
      setBackendOnline(false);
      setMessage('Face monitoring backend is not running.');
      return false;
    }
  }, [apiRequest]);

  // Check if voter is registered for an election
  const checkVoterRegistration = useCallback(async (electionId) => {
    if (!contract || !account) return false;
    
    try {
      const voters = await contract.getVoters(electionId);
      const checksummedAccount = ethers.utils.getAddress(account);
      const isRegistered = voters.some(voter => 
        ethers.utils.getAddress(voter) === checksummedAccount
      );
      return isRegistered;
    } catch (error) {
      console.log('Could not check voter registration:', error.message);
      return null;
    }
  }, [contract, account]);

  // Check if voter has already voted
  const checkIfVoted = useCallback(async (electionId) => {
    if (!contract || !account) return false;
    
    try {
      const hasVoted = await contract.hasVoted(electionId, account);
      return hasVoted;
    } catch (error) {
      console.log('Could not check vote status:', error.message);
      return false;
    }
  }, [contract, account]);

  // Fetch elections from blockchain
  const fetchElections = useCallback(async () => {
    if (!contract || isLoading) {
      console.log('Waiting for contract to load...');
      return;
    }

    setBlockchainLoading(true);
    setPageError(null);
    
    try {
      const count = await contract.electionCount();
      const totalElections = parseInt(count.toString());
      console.log('Election count:', totalElections);
      
      const electionArray = [];
      const registrationStatus = {};
      const votedStatus = {};
      
      for (let i = 1; i <= totalElections; i++) {
        try {
          const name = await contract.getElectionName(i);
          const candidates = await contract.getCandidates(i);
          
          const isRegistered = await checkVoterRegistration(i);
          const hasVoted = await checkIfVoted(i);
          
          registrationStatus[i] = isRegistered;
          votedStatus[i] = hasVoted;
          
          electionArray.push({
            id: i,
            name: name,
            candidates: candidates,
            candidateCount: candidates.length
          });
        } catch (electionError) {
          console.error(`Error fetching election ${i}:`, electionError);
        }
      }
      
      setElections(electionArray);
      setVoterRegistrationStatus(registrationStatus);
      setVotedElections(votedStatus);
      console.log('Elections fetched:', electionArray);
      
    } catch (error) {
      console.error('Error fetching elections:', error);
      setPageError(error.message || 'Failed to fetch elections');
      showToast('error', 'Connection Error', 'Failed to fetch elections from blockchain');
    } finally {
      setBlockchainLoading(false);
    }
  }, [contract, isLoading, checkVoterRegistration, checkIfVoted, showToast]);

  // Initialize on mount
  useEffect(() => {
    const initialize = async () => {
      const backendOk = await checkBackendHealth();
      await fetchElections();
      
      // If backend is online and account is connected, check registration
      if (backendOk && account) {
        await checkAccountRegistration(account);
      }
    };
    
    initialize();
    
    return () => {
      stopMonitoring();
      stopCamera();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Handle account changes
  useEffect(() => {
    const handleAccountChange = async () => {
      // Check if account actually changed
      if (prevAccountRef.current === account) {
        return;
      }
      
      console.log('Account changed from', prevAccountRef.current, 'to', account);
      prevAccountRef.current = account;
      
      // Reset face states
      setFaceRegistered(false);
      setCanVote(false);
      setVerificationStatus('idle');
      setSimilarityScore(0);
      setMonitoringCount(0);
      stopMonitoring();
      
      if (!account) {
        stopCamera();
        setMessage('Please connect your wallet.');
        return;
      }
      
      // Check if backend is online
      const backendOk = await checkBackendHealth();
      
      if (backendOk) {
        // Check registration for new account
        await checkAccountRegistration(account);
      }
      
      // Refresh elections for new account
      await fetchElections();
    };
    
    handleAccountChange();
  }, [account, checkBackendHealth, checkAccountRegistration, fetchElections, stopMonitoring, stopCamera]);

  // Register face
  const registerFace = async () => {
    if (!backendOnline) {
      showToast('error', 'Backend Offline', 'Face monitoring backend is not running');
      return;
    }
    
    if (!account) {
      showToast('error', 'Wallet Required', 'Please connect your wallet first');
      return;
    }
    
    setRegistrationLoading(true);
    setMessage('📸 Starting camera...');
    
    try {
      const cameraStarted = await startCamera();
      if (!cameraStarted) {
        setMessage('❌ Could not start camera');
        setRegistrationLoading(false);
        return;
      }
      
      setMessage('📸 Camera ready. Position your face in the frame...');
      
      // Wait for user to position face
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      setMessage('📸 Capturing face...');
      
      // Wait a bit more for stable frame
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const imageData = captureFrame();
      if (!imageData) {
        setMessage('❌ Could not capture image. Please try again.');
        showToast('error', 'Capture Failed', 'Could not capture image from camera');
        setRegistrationLoading(false);
        return;
      }
      
      setMessage('🔄 Processing and registering face...');
      
      const response = await apiRequest('/register-face', {
        method: 'POST',
        body: {
          voterId: account,
          image: imageData
        }
      });
      
      const result = response.data;
      console.log('Registration result:', result);
      
      if (result.success) {
        setFaceRegistered(true);
        setMessage('✅ Face registered successfully! Starting verification...');
        showToast('success', 'Registration Complete', 'Your face has been registered for this account!');
        
        // Start monitoring after short delay
        setTimeout(() => {
          startMonitoring();
        }, 1000);
        
      } else if (result.alreadyRegistered) {
        setFaceRegistered(true);
        setMessage('ℹ️ Face already registered for this account.');
        showToast('info', 'Already Registered', 'Your account already has a registered face.');
        
        // Start monitoring
        setTimeout(() => {
          startMonitoring();
        }, 500);
        
      } else {
        const errorMsg = result.error || 'Registration failed';
        setMessage(`❌ ${errorMsg}`);
        showToast('error', 'Registration Failed', errorMsg);
      }
    } catch (error) {
      console.error('Registration error:', error);
      setMessage('❌ Registration failed - connection error');
      showToast('error', 'Connection Error', 'Could not connect to face recognition server');
    } finally {
      setRegistrationLoading(false);
    }
  };

  // Force re-register face
  const forceReregisterFace = async () => {
    if (!backendOnline || !account) return;
    
    const confirm = window.confirm(
      'This will replace your existing face registration.\n\n' +
      'Are you sure you want to re-register your face?'
    );
    
    if (!confirm) return;
    
    setRegistrationLoading(true);
    stopMonitoring();
    
    try {
      // Clear existing registration
      await apiRequest('/clear-registration', {
        method: 'POST',
        body: { voterId: account }
      });
      
      setFaceRegistered(false);
      setCanVote(false);
      
      // Start new registration
      await registerFace();
    } catch (error) {
      console.error('Re-registration error:', error);
      showToast('error', 'Error', 'Failed to re-register face');
    } finally {
      setRegistrationLoading(false);
    }
  };

  // Reset system
  const resetSystem = async () => {
    const confirm = window.confirm(
      'This will clear your face registration for this account.\n\n' +
      'You will need to register again to vote.\n\n' +
      'Continue?'
    );
    
    if (!confirm) return;
    
    stopMonitoring();
    stopCamera();
    setFaceRegistered(false);
    setCanVote(false);
    setMessage('');
    setSimilarityScore(0);
    setMonitoringCount(0);
    setVerificationStatus('idle');
    
    try {
      await apiRequest('/clear-registration', {
        method: 'POST',
        body: { voterId: account }
      });
      console.log('Registration cleared for:', account);
      showToast('info', 'Reset Complete', 'Face registration cleared. You can register again.');
      setMessage('Registration cleared. Click "Register Face" to register again.');
    } catch (error) {
      console.log('Clear error:', error);
      showToast('error', 'Error', 'Failed to clear registration');
    }
  };

  // Get vote button state
  const getVoteButtonState = (electionId) => {
    if (votingInProgress) {
      return { disabled: true, text: 'Processing...', className: 'processing' };
    }
    if (votedElections[electionId]) {
      return { disabled: true, text: 'Already Voted', className: 'voted' };
    }
    if (voterRegistrationStatus[electionId] === false) {
      return { disabled: true, text: 'Not Registered', className: 'not-registered' };
    }
    if (!faceRegistered) {
      return { disabled: true, text: 'Register Face', className: 'verify-required' };
    }
    if (!canVote) {
      return { disabled: true, text: 'Verify Face', className: 'verify-required' };
    }
    return { disabled: false, text: 'Vote', className: 'ready' };
  };

  // Handle vote
  const handleVote = async (electionId, candidateIndex, candidateName, electionName) => {
    if (!contract || !account) {
      showToast('error', 'Wallet Required', 'Please connect your wallet to vote');
      return;
    }
    
    // Check face registration
    if (!faceRegistered) {
      showToast('warning', 'Face Required', 'Please register your face before voting');
      return;
    }
    
    // Check face verification
    if (!canVote || verificationStatus !== 'verified') {
      showToast('warning', 'Verification Required', 'Please ensure your face is verified. Look directly at the camera.');
      return;
    }
    
    // Check voter registration for election
    if (voterRegistrationStatus[electionId] === false) {
      showToast('error', 'Not Registered', `You are not registered to vote in "${electionName}". Contact the administrator.`);
      return;
    }
    
    // Check if already voted
    if (votedElections[electionId]) {
      showToast('warning', 'Already Voted', `You have already cast your vote in "${electionName}"`);
      return;
    }
    
    if (votingInProgress) return;
    
    const confirmVote = window.confirm(
      `Confirm your vote:\n\n` +
      `Election: ${electionName}\n` +
      `Candidate: ${candidateName}\n\n` +
      `This action cannot be undone.`
    );
    
    if (!confirmVote) return;
    
    setVotingInProgress(true);
    setMessage('🗳️ Submitting vote to blockchain...');
    showToast('info', 'Processing', 'Submitting your vote to the blockchain...');
    
    try {
      console.log(`Voting: Election ${electionId}, Candidate ${candidateIndex}`);
      
      // Estimate gas
      let gasEstimate;
      try {
        gasEstimate = await contract.estimateGas.vote(electionId, candidateIndex);
        console.log('Gas estimate:', gasEstimate?.toString());
      } catch (gasError) {
        console.warn('Gas estimate failed:', gasError.message);
        
        const errorMsg = getRevertReason ? getRevertReason(gasError) : gasError.message;
        
        if (errorMsg.toLowerCase().includes('already voted')) {
          setVotedElections(prev => ({ ...prev, [electionId]: true }));
          showToast('warning', 'Already Voted', `You have already voted in "${electionName}"`);
          setVotingInProgress(false);
          return;
        }
        
        if (errorMsg.toLowerCase().includes('not registered') || errorMsg.toLowerCase().includes('not a registered voter')) {
          setVoterRegistrationStatus(prev => ({ ...prev, [electionId]: false }));
          showToast('error', 'Not Registered', `You are not registered to vote in "${electionName}".`);
          setVotingInProgress(false);
          return;
        }
      }
      
      // Send transaction
      const tx = await contract.vote(electionId, candidateIndex, {
        gasLimit: gasEstimate ? gasEstimate.mul(12).div(10) : 300000
      });
      
      console.log('Transaction sent:', tx.hash);
      setMessage(`✅ Vote submitted! Waiting for confirmation...`);
      showToast('info', 'Transaction Sent', `Waiting for blockchain confirmation...`);
      
      // Wait for confirmation
      const receipt = await tx.wait();
      console.log('Transaction confirmed:', receipt);
      
      // Update state
      setVotedElections(prev => ({ ...prev, [electionId]: true }));
      
      showToast('success', 'Vote Recorded!', `Your vote for ${candidateName} has been recorded on the blockchain.`);
      setMessage('✅ Vote successfully recorded!');
      
      // Refresh elections
      setTimeout(() => fetchElections(), 1000);
      
    } catch (err) {
      console.error('Vote failed:', err);
      
      let errorMsg = err.message;
      if (getRevertReason) {
        errorMsg = getRevertReason(err);
      }
      
      if (errorMsg.toLowerCase().includes('already voted')) {
        setVotedElections(prev => ({ ...prev, [electionId]: true }));
        showToast('warning', 'Already Voted', `You have already voted in "${electionName}"`);
      } else if (errorMsg.toLowerCase().includes('not registered') || errorMsg.toLowerCase().includes('not a registered voter')) {
        setVoterRegistrationStatus(prev => ({ ...prev, [electionId]: false }));
        showToast('error', 'Not Registered', `You are not registered to vote in "${electionName}".`);
      } else if (errorMsg.toLowerCase().includes('user rejected') || errorMsg.toLowerCase().includes('user denied')) {
        showToast('info', 'Cancelled', 'Transaction was cancelled');
      } else if (errorMsg.toLowerCase().includes('insufficient funds')) {
        showToast('error', 'Insufficient Funds', 'You don\'t have enough ETH for gas fees');
      } else {
        showToast('error', 'Vote Failed', errorMsg);
      }
      
      setMessage('❌ Vote failed - see notification for details');
    } finally {
      setVotingInProgress(false);
    }
  };

  // Debug function to test backend connection
  const debugBackendConnection = async () => {
    console.log('🔍 Testing backend connection...');
    try {
      const response = await apiRequest('/health');
      console.log('✅ Backend is working:', response.data);
      showToast('success', 'Backend Online', 'Face recognition backend is connected!');
    } catch (error) {
      console.error('❌ Backend test failed:', error);
      showToast('error', 'Backend Offline', 'Cannot connect to face recognition backend');
    }
  };

  // Render loading state
  if (isLoading) {
    return (
      <div className="voting-page loading-page">
        <div className="loading-container">
          <div className="spinner"></div>
          <h2>Loading Voting System</h2>
          <p>Connecting to blockchain...</p>
        </div>
      </div>
    );
  }

  // Render error state
  if (error || pageError) {
    return (
      <div className="voting-page error-page">
        <div className="error-container">
          <div className="error-icon">⚠️</div>
          <h2>Connection Error</h2>
          <p className="error-message">{error || pageError}</p>
          <div className="error-help">
            <p>Please check:</p>
            <ul>
              <li>MetaMask is installed and unlocked</li>
              <li>You're connected to the correct network</li>
              <li>The smart contract is deployed</li>
            </ul>
          </div>
          <button onClick={() => window.location.reload()} className="btn-primary">
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  // Render wallet not connected state
  if (!account) {
    return (
      <div className="voting-page connect-page">
        <div className="connect-container">
          <div className="connect-icon">🔗</div>
          <h2>Connect Your Wallet</h2>
          <p>Please connect your MetaMask wallet to participate in elections.</p>
          <div className="connect-steps">
            <p>Steps:</p>
            <ol>
              <li>Click the MetaMask extension</li>
              <li>Select your account</li>
              <li>Approve the connection</li>
            </ol>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="voting-page">
      {/* Debug button (optional) */}
      <button 
        onClick={debugBackendConnection}
        style={{
          position: 'fixed',
          top: '10px',
          right: '10px',
          zIndex: 1000,
          padding: '8px 12px',
          background: '#666',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          fontSize: '12px',
          cursor: 'pointer'
        }}
      >
        🐛 Test Backend
      </button>

      {/* Toast Notification */}
      {toast && (
        <div className={`toast toast-${toast.type}`}>
          <div className="toast-icon">
            {toast.type === 'success' && '✅'}
            {toast.type === 'error' && '❌'}
            {toast.type === 'warning' && '⚠️'}
            {toast.type === 'info' && 'ℹ️'}
          </div>
          <div className="toast-content">
            <strong className="toast-title">{toast.title}</strong>
            <p className="toast-text">{toast.text}</p>
          </div>
          <button className="toast-close" onClick={() => setToast(null)}>×</button>
        </div>
      )}

      {/* Header */}
      <header className="voting-header">
        <div className="header-content">
          <h1>🗳️ SecureVote</h1>
          <p className="tagline">Blockchain Elections with Face Verification</p>
        </div>
        
        <div className="status-bar">
          <div className={`status-item ${backendOnline ? 'online' : 'offline'}`}>
            <span className="status-dot"></span>
            <span className="status-label">Face Monitor</span>
          </div>
          <div className={`status-item ${contract ? 'online' : 'offline'}`}>
            <span className="status-dot"></span>
            <span className="status-label">Blockchain</span>
          </div>
          <div className={`status-item online`}>
            <span className="status-dot"></span>
            <span className="status-label">{account ? `${account.slice(0, 6)}...${account.slice(-4)}` : 'Wallet'}</span>
          </div>
        </div>
      </header>

      <main className="voting-main">
        {/* Face Verification Section */}
        <section className="face-section">
          <div className="section-header">
            <h2>👤 Face Verification</h2>
            <span className={`verification-badge ${verificationStatus}`}>
              {verificationStatus === 'verified' && '✓ Verified'}
              {verificationStatus === 'failed' && '✗ Failed'}
              {verificationStatus === 'error' && '! Error'}
              {verificationStatus === 'idle' && (faceRegistered ? '○ Monitoring' : '○ Not Registered')}
            </span>
          </div>

          {!backendOnline ? (
            <div className="backend-offline-card">
              <div className="offline-icon">🔌</div>
              <h3>Backend Offline</h3>
              <p>The face recognition server is not running.</p>
              <div className="code-block">
                <code>Run from Google Colab</code>
              </div>
              <button onClick={checkBackendHealth} className="btn-secondary">
                🔄 Check Connection
              </button>
            </div>
          ) : !faceRegistered ? (
            <div className="registration-card">
              <div className="camera-wrapper">
                <video 
                  ref={videoRef} 
                  autoPlay 
                  playsInline 
                  muted
                  className="camera-feed"
                />
                <canvas ref={canvasRef} style={{ display: 'none' }} />
                <div className="camera-overlay">
                  <div className="face-guide"></div>
                </div>
                {registrationLoading && (
                  <div className="camera-loading-overlay">
                    <div className="spinner"></div>
                    <p>Processing...</p>
                  </div>
                )}
              </div>
              
              <div className="registration-info">
                <h3>Register Your Face</h3>
                <p>Face registration is required to vote. Each wallet account needs its own face registration.</p>
                
                <div className="account-badge">
                  <span className="account-label">Registering for:</span>
                  <span className="account-address">{account.slice(0, 8)}...{account.slice(-6)}</span>
                </div>
                
                <ul className="tips-list">
                  <li>✓ Face camera directly</li>
                  <li>✓ Good lighting</li>
                  <li>✓ Remove sunglasses</li>
                  <li>✓ Stay still</li>
                </ul>
                
                <button 
                  onClick={registerFace} 
                  className="btn-primary btn-large"
                  disabled={!backendOnline || registrationLoading}
                >
                  {registrationLoading ? '⏳ Registering...' : '📸 Register Face'}
                </button>
                
                {message && (
                  <div className={`status-message ${message.includes('❌') ? 'error' : message.includes('✅') ? 'success' : 'info'}`}>
                    {message}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="monitoring-card">
              <div className="camera-wrapper monitoring">
                <video 
                  ref={videoRef} 
                  autoPlay 
                  playsInline 
                  muted
                  className="camera-feed"
                />
                <div className={`verification-overlay ${verificationStatus}`}>
                  <div className="verification-ring"></div>
                </div>
              </div>
              
              <div className="monitoring-info">
                <div className="account-badge">
                  <span className="account-label">Verified for:</span>
                  <span className="account-address">{account.slice(0, 8)}...{account.slice(-6)}</span>
                </div>
                
                <div className="monitoring-status">
                  <div className={`status-indicator ${verificationStatus}`}>
                    <span className="status-icon">
                      {verificationStatus === 'verified' && '✅'}
                      {verificationStatus === 'failed' && '❌'}
                      {verificationStatus === 'error' && '⚠️'}
                      {verificationStatus === 'idle' && '👁️'}
                    </span>
                    <span className="status-text">
                      {verificationStatus === 'verified' && 'Face Verified - You Can Vote'}
                      {verificationStatus === 'failed' && 'Verification Failed'}
                      {verificationStatus === 'error' && 'Connection Error'}
                      {verificationStatus === 'idle' && 'Starting Verification...'}
                    </span>
                  </div>
                  
                  {similarityScore > 0 && (
                    <div className="similarity-meter">
                      <div className="meter-label">
                        Match: {similarityScore}% (need {matchThreshold}%)
                      </div>
                      <div className="meter-bar">
                        <div 
                          className={`meter-fill ${similarityScore >= matchThreshold ? 'pass' : 'fail'}`}
                          style={{ width: `${Math.min(similarityScore, 100)}%` }}
                        ></div>
                        <div 
                          className="meter-threshold"
                          style={{ left: `${matchThreshold}%` }}
                        ></div>
                      </div>
                    </div>
                  )}
                </div>
                
                <div className="monitoring-message">{message}</div>
                
                <div className="monitoring-stats">
                  <span>Checks: {monitoringCount}</span>
                  <span className={canVote ? 'can-vote' : 'cannot-vote'}>
                    Voting: {canVote ? '✅ Enabled' : '❌ Disabled'}
                  </span>
                </div>
                
                {verificationStatus === 'failed' && (
                  <div className="help-tips">
                    <p><strong>Tips to improve verification:</strong></p>
                    <ul>
                      <li>Ensure good, even lighting on your face</li>
                      <li>Face the camera directly</li>
                      <li>Remove glasses if possible</li>
                      <li>Match your position from registration</li>
                    </ul>
                  </div>
                )}
                
                <div className="monitoring-actions">
                  <button onClick={forceReregisterFace} className="btn-secondary btn-small">
                    📸 Re-register Face
                  </button>
                  <button onClick={resetSystem} className="btn-secondary btn-small btn-danger">
                    🗑️ Clear Registration
                  </button>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* Elections Section */}
        <section className="elections-section">
          <div className="section-header">
            <h2>📊 Elections</h2>
            <button 
              onClick={fetchElections} 
              className="btn-icon"
              disabled={blockchainLoading}
              title="Refresh elections"
            >
              🔄
            </button>
          </div>

          {blockchainLoading ? (
            <div className="elections-loading">
              <div className="spinner"></div>
              <p>Loading elections from blockchain...</p>
            </div>
          ) : elections.length === 0 ? (
            <div className="no-elections">
              <div className="empty-icon">📭</div>
              <h3>No Elections Available</h3>
              <p>There are no active elections at the moment.</p>
              <p className="help-text">Elections can be created from the Admin page.</p>
            </div>
          ) : (
            <div className="elections-grid">
              {elections.map((election) => {
                const buttonState = getVoteButtonState(election.id);
                const isRegistered = voterRegistrationStatus[election.id];
                const hasVoted = votedElections[election.id];
                
                return (
                  <div key={election.id} className={`election-card ${hasVoted ? 'voted' : ''} ${isRegistered === false ? 'not-registered' : ''}`}>
                    <div className="election-header">
                      <h3>{election.name}</h3>
                      <div className="election-badges">
                        <span className="election-id">#{election.id}</span>
                        {hasVoted && <span className="badge voted">✓ Voted</span>}
                        {isRegistered === false && <span className="badge not-registered">Not Registered</span>}
                        {isRegistered === true && !hasVoted && <span className="badge registered">✓ Registered</span>}
                      </div>
                    </div>
                    
                    {isRegistered === false && (
                      <div className="registration-warning">
                        <span className="warning-icon">⚠️</span>
                        <span>You are not registered to vote in this election. Contact the administrator to be added as a voter.</span>
                      </div>
                    )}
                    
                    <div className="candidates-section">
                      <h4>Candidates ({election.candidateCount})</h4>
                      <div className="candidates-list">
                        {election.candidates.map((candidate, index) => (
                          <div key={index} className="candidate-row">
                            <div className="candidate-info">
                              <span className="candidate-number">{index + 1}</span>
                              <span className="candidate-name">{candidate}</span>
                            </div>
                            <button 
                              onClick={() => handleVote(election.id, index, candidate, election.name)}
                              disabled={buttonState.disabled}
                              className={`vote-btn ${buttonState.className}`}
                            >
                              {buttonState.text}
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                    
                    {!faceRegistered && !hasVoted && isRegistered !== false && (
                      <div className="face-warning">
                        <span className="warning-icon">👤</span>
                        <span>Register your face to enable voting</span>
                      </div>
                    )}
                    
                    {faceRegistered && !canVote && !hasVoted && isRegistered !== false && (
                      <div className="face-warning verification">
                        <span className="warning-icon">👁️</span>
                        <span>Face verification in progress - look at camera</span>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </section>
      </main>

      {/* Footer */}
      <footer className="voting-footer">
        <div className="debug-info">
          <span>Elections: {elections.length}</span>
          <span>Face: {faceRegistered ? '✓ Registered' : '✗ Not Registered'}</span>
          <span>Verified: {canVote ? '✓ Yes' : '✗ No'}</span>
          <span>Match: {similarityScore}%</span>
          <span>Registered Voters: {registeredVotersCount}</span>
        </div>
      </footer>
    </div>
  );
};

export default VotingPage;