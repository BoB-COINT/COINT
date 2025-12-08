import { useState, useRef } from 'react';
import searchIcon from '../assets/icons/search.png';
import { useNavigate  } from 'react-router-dom';

function SearchBar() {  
    const [TokenAddr, setTokenAddr] = useState('');
    const [loading, setLoading] = useState(false);
    const inputRef = useRef(null);
    const navigate = useNavigate();

    const handleSearch = () => {
        const addr = (TokenAddr || '').trim();
        if (!addr) return;

        setLoading(true);
        navigate(`/loading?address=${addr}`);
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    };

    return (
        <div style={{
            padding : '30px',
            color : '#171821',
            display: 'flex',
            gap: '10px'
        }}>
        <input
            ref={inputRef}
            value={TokenAddr}
            onChange={(e) => setTokenAddr(e.target.value)}
            onKeyDown={handleKeyPress}
            spellCheck={false}
            autoCapitalize="none"
            autoCorrect="off"
            placeholder="Search here.."
            className="search-input" 
            style={{
                flex: '1',
                padding: '15px',
                borderRadius: '999px',  
                border: '2px solid rgba(255, 255, 255, 0.08)',
                backgroundColor: 'rgba(33, 34, 45, 0.75)',
                backdropFilter: 'blur(12px)',
                WebkitBackdropFilter: 'blur(12px)',
                color: '#d2d2d2'
            }}
        />
        <button
            type="button"
            onClick={handleSearch}
            disabled={loading}
            style={{
                padding: '10px 18px',
                borderRadius: '999px',
                border: '2px solid rgba(255, 255, 255, 0.08)',
                backgroundColor: 'rgba(33, 34, 45, 0.85)',
                backdropFilter: 'blur(12px)',
                WebkitBackdropFilter: 'blur(12px)',
                cursor: loading ? 'progress' : 'pointer',
                opacity: loading ? 0.7 : 1,
                display: 'flex',
                alignItems: 'center',
            }}
        >
            <img
            src={searchIcon}
            alt={loading ? 'Analyzing...' : 'search'}
            style={{
                width: '15px',
                height: '15px',
                filter: 'brightness(0) invert(1)'
            }}
            />
        </button>


    </div>
    )

}

export default SearchBar;