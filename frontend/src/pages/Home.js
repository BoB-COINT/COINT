import './Home.css';
import SearchBar from '../components/SearchBar';
import logo from '../logo.svg'; 

function Home() {
  return (
    <div className="home-search-wrapper">
        <div className="home-search-inner">
            <img src={logo} alt="COINT" className="home-logo" />

            <p className="home-tagline">
            <span className="home-tagline-em">
                Just enter a Token Address
            </span>
            {" — we detect honeypots, exit scams,"}
            <br />
            {"and abnormal patterns, then turn them into shared security intelligence."}
            </p>

            <SearchBar />
        </div>
    </div>
  );
}

export default Home;