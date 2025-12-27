import React from 'react';
import { Link } from 'react-router-dom';
import './HomePage.css';
import { 
    Shield, 
    Vote, 
    Users, 
    Lock, 
    CheckCircle, 
    TrendingUp,
    Award,
    Clock,
    BarChart3,
    Zap
} from 'lucide-react';
const HomePage = () => {
    const features = [
        {
            icon: <Shield className="feature-icon" />,
            title: "Blockchain Secured",
            description: "Every vote is cryptographically secured and immutable on the blockchain"
        },
        {
            icon: <Lock className="feature-icon" />,
            title: "Anonymous Voting",
            description: "Maintain voter privacy while ensuring transparency in results"
        },
        {
            icon: <CheckCircle className="feature-icon" />,
            title: "Instant Verification",
            description: "Voters can verify their vote was counted without revealing their choice"
        },
        {
            icon: <BarChart3 className="feature-icon" />,
            title: "Real-time Results",
            description: "Live updates and transparent counting process visible to all"
        }
    ];

    const stats = [
        { number: "10K+", label: "Votes Cast" },
        { number: "50+", label: "Elections Held" },
        { number: "99.9%", label: "Uptime" },
        { number: "0", label: "Security Breaches" }
    ];

    return (
        <div className="homepage">
            {/* Hero Section */}
            <section className="hero-section">
                <div className="hero-content">
                    <div className="hero-badge">
                        <Zap size={16} />
                        <span>Powered by Blockchain</span>
                    </div>
                    <h1 className="hero-title">
                        Digital Democracy, <br />
                        <span className="hero-highlight">Secured Forever</span>
                    </h1>
                    <p className="hero-description">
                        Transform your elections with blockchain-powered voting. 
                        Transparent, secure, and tamper-proof elections for colleges, 
                        organizations, and communities.
                    </p>
                    <div className="hero-actions">
                        <Link to="/vote" className="btn btn-primary">
                            <Vote size={20} />
                            Start Voting
                        </Link>
                        <Link to="/demo" className="btn btn-secondary">
                            Watch Demo
                        </Link>
                    </div>
                </div>
                <div className="hero-image">
                    <div className="floating-card">
                        <Clock size={24} />
                        <div>
                            <strong>Next Election</strong>
                            <p>Student Council 2024</p>
                        </div>
                    </div>
                    <div className="floating-card delayed">
                        <Users size={24} />
                        <div>
                            <strong>Active Voters</strong>
                            <p>2,847 registered</p>
                        </div>
                    </div>
                </div>
            </section>

            {/* Features Section */}
            <section className="features-section">
                <h2 className="section-title">Why Choose SecureVote?</h2>
                <div className="features-grid">
                    {features.map((feature, index) => (
                        <div key={index} className="feature-card">
                            {feature.icon}
                            <h3>{feature.title}</h3>
                            <p>{feature.description}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* Stats Section */}
            <section className="stats-section">
                <div className="stats-container">
                    {stats.map((stat, index) => (
                        <div key={index} className="stat-item">
                            <div className="stat-number">{stat.number}</div>
                            <div className="stat-label">{stat.label}</div>
                        </div>
                    ))}
                </div>
            </section>

            {/* CTA Cards Section */}
            <section className="cta-section">
                <div className="cta-container">
                    <div className="cta-card admin-card">
                        <Award className="cta-icon" />
                        <h3>Election Administrators</h3>
                        <p>
                            Create and manage elections with complete control. 
                            Register voters, set up ballots, and monitor the entire 
                            process with our intuitive dashboard.
                        </p>
                        <ul className="cta-features">
                            <li>✓ Easy voter registration</li>
                            <li>✓ Custom ballot creation</li>
                            <li>✓ Real-time monitoring</li>
                            <li>✓ Detailed analytics</li>
                        </ul>
                        <Link to="/admin" className="cta-button">
                            Access Admin Portal
                        </Link>
                    </div>

                    <div className="cta-card voter-card">
                        <Vote className="cta-icon" />
                        <h3>Voters</h3>
                        <p>
                            Cast your vote securely from any device. Your vote is 
                            encrypted, anonymous, and permanently recorded on the 
                            blockchain.
                        </p>
                        <ul className="cta-features">
                            <li>✓ Simple voting process</li>
                            <li>✓ Vote verification</li>
                            <li>✓ Complete anonymity</li>
                            <li>✓ Instant confirmation</li>
                        </ul>
                        <Link to="/vote" className="cta-button">
                            Enter Voting Booth
                        </Link>
                    </div>
                </div>
            </section>

            {/* Trust Section */}
            <section className="trust-section">
                <h2>Trusted by Leading Institutions</h2>
                <p className="trust-description">
                    Join colleges and organizations already using SecureVote for their elections
                </p>
                <div className="trust-logos">
                    {/* Add actual logos or placeholder divs */}
                    <div className="trust-logo">LBRCE</div>
                    <div className="trust-logo">Tech Council</div>
                    <div className="trust-logo">Student Union</div>
                    <div className="trust-logo">AIDS Department</div>
                </div>
            </section>

            {/* Quick Links Footer */}
            <section className="quick-actions">
                <div className="quick-action-items">
                    <Link to="/results" className="quick-link">
                        <TrendingUp size={20} />
                        View Past Results
                    </Link>
                    <Link to="/verify" className="quick-link">
                        <CheckCircle size={20} />
                        Verify Your Vote
                    </Link>
                    <Link to="/help" className="quick-link">
                        <Users size={20} />
                        Get Support
                    </Link>
                </div>
            </section>
        </div>
    );
};

export default HomePage;