import { createContext, useContext, useEffect, useState } from 'react';
import { ethers } from 'ethers';

export const Web3Context = createContext();

// Improved revert reason extractor for ethers.js
const getRevertReason = (error) => {
  if (!error) return 'Unknown error occurred';
  
  console.log('Error details:', error);
  
  // User rejection from MetaMask
  if (error.code === 4001 || error.code === 'ACTION_REJECTED' || 
      (error.message && error.message.toLowerCase().includes('user denied'))) {
    return 'Transaction rejected by user';
  }

  const errorMessage = error.message || '';
  
  // Common contract revert patterns
  const patterns = [
    /reverted with reason string ['"]([^'"]+)['"]/i,
    /execution reverted: ['"]([^'"]+)['"]/i,
    /reverted: ['"]([^'"]+)['"]/i,
    /reason: ['"]([^'"]+)['"]/i,
    /Only owner can call this function/i,
    /Voter is already registered/i,
    /You have already voted in this election/i,
    /You are not registered to vote in this election/i,
    /Invalid candidate index/i,
    /Election does not exist/i,
    /Provide at least one candidate/i
  ];
  
  for (const pattern of patterns) {
    const match = errorMessage.match(pattern);
    if (match) {
      if (match[1]) return match[1];
      return match[0];
    }
  }

  // Ethers.js specific error structure
  if (error.reason) {
    return error.reason;
  }
  
  if (error.data && error.data.message) {
    return error.data.message;
  }

  // CALL_EXCEPTION errors
  if (error.code === 'CALL_EXCEPTION') {
    return 'Contract call failed. Make sure contract exists and you have correct permissions.';
  }

  // Gas errors
  if (errorMessage.includes('gas') || errorMessage.includes('out of gas')) {
    return 'Transaction ran out of gas. Try increasing gas limit.';
  }

  // Return actual error message
  if (errorMessage) {
    return errorMessage.length > 150 ? errorMessage.substring(0, 150) + '...' : errorMessage;
  }

  return 'Transaction failed. Check console for details.';
};

export const Web3Provider = ({ children }) => {
  const [provider, setProvider] = useState(null);
  const [signer, setSigner] = useState(null);
  const [contract, setContract] = useState(null);
  const [account, setAccount] = useState(null);
  const [owner, setOwner] = useState(null);
  const [isAdmin, setIsAdmin] = useState(false);
  const [network, setNetwork] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Network check function
  const checkAndSwitchNetwork = async () => {
    try {
      if (!window.ethereum) return false;
      
      const chainId = await window.ethereum.request({ method: 'eth_chainId' });
      const chainIdNum = parseInt(chainId, 16);
      
      // Accept Hardhat networks (1337 or 31337)
      if (chainIdNum === 1337 || chainIdNum === 31337) {
        console.log('✅ Connected to Hardhat network');
        return true;
      }
      
      console.log('❌ Wrong network. Current:', chainIdNum, '(Should be 1337)');
      
      // Try to switch to Hardhat (1337 in hex = 0x539)
      try {
        await window.ethereum.request({
          method: 'wallet_switchEthereumChain',
          params: [{ chainId: '0x539' }],
        });
        console.log('✅ Switched to Hardhat network');
        return true;
      } catch (switchError) {
        // Network not added, add it
        if (switchError.code === 4902) {
          await window.ethereum.request({
            method: 'wallet_addEthereumChain',
            params: [{
              chainId: '0x539',
              chainName: 'Hardhat Local',
              rpcUrls: ['http://127.0.0.1:8545'],
              nativeCurrency: {
                name: 'ETH',
                symbol: 'ETH',
                decimals: 18
              }
            }],
          });
          console.log('✅ Added Hardhat network');
          return true;
        }
        throw switchError;
      }
    } catch (error) {
      console.error('Network switch failed:', error);
      return false;
    }
  };

  const initWeb3 = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      console.log('🚀 Initializing Web3...');
      
      if (!window.ethereum) {
        setError('MetaMask not found');
        setIsLoading(false);
        alert('Please install MetaMask to use this application!');
        return;
      }

      // Ensure correct network
      const networkOk = await checkAndSwitchNetwork();
      if (!networkOk) {
        setError('Please switch MetaMask to Hardhat network (Chain ID: 1337)');
        setIsLoading(false);
        return;
      }

      // Initialize ethers provider
      const provider = new ethers.providers.Web3Provider(window.ethereum);
      setProvider(provider);
      
      // Get network info
      const networkInfo = await provider.getNetwork();
      setNetwork(networkInfo);
      console.log('🌐 Connected to network:', networkInfo.name, 'Chain ID:', networkInfo.chainId);

      // Request accounts
      const accounts = await provider.send('eth_requestAccounts', []);
      const acct = accounts[0];
      setAccount(acct);
      console.log('👤 Account:', acct);

      // Get contract address
      const contractAddress = window.__CONTRACT_ADDRESS;
      if (!contractAddress) {
        setError('Contract address not loaded. Please refresh page.');
        setIsLoading(false);
        console.error('❌ Contract address is undefined');
        return;
      }
      console.log('📄 Contract Address:', contractAddress);

      // Verify contract exists
      const code = await provider.getCode(contractAddress);
      if (code === '0x' || code === '0x0') {
        setError(`No contract deployed at address: ${contractAddress}\nPlease redeploy the contract.`);
        setIsLoading(false);
        console.error('❌ No contract code at address');
        return;
      }
      console.log('✅ Contract code verified, length:', code.length);

      // Load ABI - try multiple methods
      let abi;
      try {
        // Try from artifacts
        const artifact = require('../artifacts/contracts/Voting.sol/Voting.json');
        abi = artifact.abi;
        console.log('✅ ABI loaded via require');
      } catch (error) {
        console.error('ABI load failed:', error);
        // Fallback minimal ABI
        abi = [
          {
            "inputs": [],
            "name": "owner",
            "outputs": [{"internalType": "address", "name": "", "type": "address"}],
            "stateMutability": "view",
            "type": "function"
          },
          {
            "inputs": [],
            "name": "electionCount",
            "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function"
          }
        ];
        console.log('⚠️ Using minimal ABI fallback');
      }

      // Create contract instance
      const signer = provider.getSigner();
      setSigner(signer);
      const votingContract = new ethers.Contract(contractAddress, abi, signer);
      setContract(votingContract);

      // Test contract and get owner
      try {
        const electionCount = await votingContract.electionCount();
        console.log('✅ electionCount:', electionCount.toString());
        
        const contractOwner = await votingContract.owner();
        setOwner(contractOwner);
        console.log('✅ Contract owner:', contractOwner);
        
        const adminStatus = acct.toLowerCase() === contractOwner.toLowerCase();
        setIsAdmin(adminStatus);
        console.log('👑 Is admin?', adminStatus);
        
        console.log('🎉 Web3 initialized successfully!');
        
      } catch (testError) {
        console.error('❌ Contract test failed:', testError);
        setError(`Contract call failed: ${testError.message}`);
      }

    } catch (err) {
      console.error('💥 Web3 initialization failed:', err);
      setError(err.message || 'Failed to initialize Web3');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    initWeb3();

    // Set up event listeners
    if (window.ethereum) {
      const handleAccountsChanged = (accounts) => {
        console.log('Accounts changed:', accounts);
        const newAcct = accounts[0] || null;
        setAccount(newAcct);
        if (newAcct && owner) {
          setIsAdmin(newAcct.toLowerCase() === owner.toLowerCase());
        }
      };

      const handleChainChanged = () => {
        console.log('Network changed, reloading...');
        window.location.reload();
      };

      window.ethereum.on('accountsChanged', handleAccountsChanged);
      window.ethereum.on('chainChanged', handleChainChanged);

      return () => {
        if (window.ethereum.removeListener) {
          window.ethereum.removeListener('accountsChanged', handleAccountsChanged);
          window.ethereum.removeListener('chainChanged', handleChainChanged);
        }
      };
    }
  }, []);

  useEffect(() => {
    if (account && owner) {
      setIsAdmin(account.toLowerCase() === owner.toLowerCase());
    } else {
      setIsAdmin(false);
    }
  }, [account, owner]);

  const connectWallet = async () => {
    await initWeb3();
  };

  return (
    <Web3Context.Provider value={{ 
      provider,
      signer,
      contract, 
      account, 
      owner, 
      isAdmin, 
      network,
      isLoading,
      error,
      getRevertReason,
      connectWallet,
      initWeb3
    }}>
      {children}
    </Web3Context.Provider>
  );
};

export const useWeb3 = () => useContext(Web3Context);