import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import HomePage from './components/HomePage';
import AdminPage from './components/AdminPage';
import VotingPage from './components/VotingPage';
import HowToUsePage from './components/HowToUsePage';
import { Web3Provider, useWeb3 } from './contexts/Web3Context';
import './App.css';

// Navbar consumes context (must be inside provider)
const NavBar = () => {
  const { isAdmin, owner } = useWeb3();
  return (
    <header className="app-header">
      <Link to="/" className="logo">SecureVote</Link>
      <nav>
        <Link to="/">Home</Link>
        {isAdmin && <Link to="/admin">Admin Dashboard</Link>}
        <Link to="/vote">Voting Booth</Link>
        <Link to="/how-to-use">How to Use</Link>
      </nav>
      <div style={{ marginLeft: 16, fontSize: 12, color: '#666' }}>
        Owner: {owner ? owner : 'not loaded'}
      </div>
    </header>
  );
};

function App() {
  return (
    <Web3Provider>
      <Router>
        <div className="App">
          <NavBar />
          <main>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/admin" element={<AdminPage />} />
              <Route path="/vote" element={<VotingPage />} />
              <Route path="/how-to-use" element={<HowToUsePage />} />
            </Routes>
          </main>
        </div>
      </Router>
    </Web3Provider>
  );
}

export default App;