import React from 'react';
import { Link } from 'react-router-dom';

const HomePage = () => {
    return (
        <div className="page-container">
            <h2>Welcome to the Digital Voting Platform</h2>
            <p className="page-description">
                A secure and transparent voting system for your community, powered by blockchain technology. 
                Perfect for college elections, class representatives, and community polls.
            </p>
            <div className="card-container">
                <div className="card">
                    <h3>For Election Administrators</h3>
                    <p>Set up your election, register eligible voters, and ensure a fair process from start to finish.</p>
                    <Link to="/admin" className="button">Admin Dashboard</Link>
                </div>
                <div className="card">
                    <h3>For Voters</h3>
                    <p>Cast your vote with confidence, knowing your choice is secure and the results are transparent and tamper-proof.</p>
                    <Link to="/vote" className="button">Go to Voting Booth</Link>
                </div>
            </div>
        </div>
    );
};

export default HomePage;