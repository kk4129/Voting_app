import { useEffect, useState } from 'react';
import Web3 from 'web3';
import VotingContract from '../contracts/Voting.json';

const useWeb3 = () => {
    const [account, setAccount] = useState(null);
    const [contract, setContract] = useState(null);
    const [web3, setWeb3] = useState(null);

    useEffect(() => {
        const initWeb3 = async () => {
            if (window.ethereum) {
                const web3Instance = new Web3(window.ethereum);
                await window.ethereum.request({ method: 'eth_requestAccounts' });
                const accounts = await web3Instance.eth.getAccounts();
                const networkId = await web3Instance.eth.net.getId();
                const deployedNetwork = VotingContract.networks[networkId];
                const contractInstance = new web3Instance.eth.Contract(
                    VotingContract.abi,
                    deployedNetwork && deployedNetwork.address,
                );

                setWeb3(web3Instance);
                setAccount(accounts[0]);
                setContract(contractInstance);
            } else {
                console.error('Please install MetaMask!');
            }
        };

        initWeb3();
    }, []);

    return { web3, account, contract };
};

export default useWeb3;