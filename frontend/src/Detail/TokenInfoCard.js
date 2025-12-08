import './TokenInfoCard.css';
import { useState } from 'react';

function TokenInfoCard({ token }) {
  const [copiedKey, setCopiedKey] = useState(null);  

  const copyAddr = (key, value) => {  
    navigator.clipboard.writeText(value);
    setCopiedKey(key);  
    setTimeout(() => setCopiedKey(null), 900); 
  }

  return (
    <div className="token-info-card">

      {/* 헤더 영역 */}
      <div className="card-header">
        <span className="card-title">Token Info</span>
      </div>
      
      {/* Token Name */}
      <div className="token-name-section">
        <span className="token-name">{token.tokenName}</span>
        <span className="token-symbol">{token.symbol}</span>
        <span className="addr-pill">{token.address}</span>
        <button className="icon-btn" onClick={() => copyAddr('token', token.address)}>
          {copiedKey === 'token' ? '✓' : '⧉'}
        </button>
      </div>

      <div className="info-item">
        <span className="info-label">Token Type</span>
        <span className="info-value">{token.tokenType}</span>
      </div>

      <div className="info-item">
        <span className="info-label">Contract Owner</span>
        <span className="info-value">{token.contractOwner}</span>
        <button className="icon-btn" onClick={() => copyAddr('owner', token.contractOwner)}>
          {copiedKey === 'owner' ? '✓' : '⧉'}
        </button>   
      </div>  

      <div className="info-item">
        <span className="info-label">Pair</span>
        <span className="info-value">{token.pair}</span>
        <button className="icon-btn" onClick={() => copyAddr('pair', token.pair)}>
          {copiedKey === 'pair' ? '✓' : '⧉'}
        </button>
      </div> 
      
      <div className="info-sub">
        <span className="meta">
          <span className="meta-label">Token CreateTime</span>
          <span className="meta-paren">(</span>
          <span className="meta-value">{token.tokenCreateTs || '-'}</span>
          <span className="meta-paren">)</span>
        </span>
        <span className="meta-sep"> · </span>
        <span className="meta">
          <sapn className="meta-label">Pair CreateTime</sapn>
          <span className="meta-paren">(</span>
          <span className="meta-value">{token.pairCreateTs || '-'}</span>
          <span className="meta-paren">)</span>
        </span>
      </div>

    </div>
  );
}

export default TokenInfoCard;