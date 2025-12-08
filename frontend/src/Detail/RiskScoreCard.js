import './RiskScoreCard.css';

function RiskScoreCard({ token }) {
    const radius1 = 80;  // 바깥쪽 게이지 (크기 증가)
    const radius2 = 55;  // 안쪽 게이지 (크기 증가)
    const circumference1 = Math.PI * radius1;
    const circumference2 = Math.PI * radius2;
    
    // token 객체에서 scamTypeDistribution 가져오기
    const scamTypes = token.scamTypeDistribution || [];

    // 위험도 레벨 결정 함수
    const getRiskLevel = () => {
        const honeypot = scamTypes.find(s => s.type === 'Honeypot')?.percentage ?? 0;
        const exit = scamTypes.find(s => s.type === 'Exit')?.percentage ?? 0;

        // 어떤 스코어가 더 높은지 먼저 보고, 그 타입 기준으로 구간 적용
        const isHoneypotMax = honeypot >= exit;
        const value = isHoneypotMax ? honeypot : exit;

        const ranges = isHoneypotMax
            ? { safe: 2.00, caution: 48.999999999999977, warning: 97.9 }          // Honeypot
            : { safe: 2.00, caution: 78.0502200126648, warning: 99.5 };           // Exit

        if (value < ranges.safe)   return { level: 'Safe',     color: '#00C853' };
        if (value < ranges.caution) return { level: 'Caution',  color: '#FFC107' };
        if (value < ranges.warning) return { level: 'Warning',  color: '#FF9500' };
        return { level: 'Critical', color: '#FF4444' };
    };

    // 스코어(0~1)와 타입에 따른 색상 결정
    const getColorByPercentage = (type, score) => {
        const value = typeof score === 'number' ? score : 0;

        const ranges = type === 'Honeypot'
            ? { safe: 2.00, caution: 48.999999999999977, warning: 97.9 }          // Honeypot 구간
            : { safe: 2.00, caution: 78.0502200126648, warning: 99.5 };           // Exit 구간

        if (value < ranges.safe)   return '#00C853'; // Safe
        if (value < ranges.caution) return '#FFC107'; // Caution
        if (value < ranges.warning) return '#FF9500'; // Warning
        return '#FF4444';                              // Critical
    };

    const getOffset = (percentage, circumference) => {
        return circumference - (percentage / 100) * circumference;
    };

    const getLevelColor = (level) => {
        switch(level) {
            case 'Critical': return '#FF4444';
            case 'Warning':  return '#FF9500';
            case 'Caution':  return '#FFC107';
            case 'Safe':     return '#00C853';
            default:         return '#FFFFFF';
        }
    };
    const riskLevel = getRiskLevel();

    return (
        <div className="risk-score-card">
            <h3 className="card-title">Risk Score</h3>

            {/* 위험도 레벨 표시 */}
            <div className="risk-level-display">
                <span 
                    className="risk-level-text"
                    style={{ color: riskLevel.color }}
                >
                    {riskLevel.level}
                </span>
            </div>

            <div className="combined-gauge-container">
                <svg width="260" height="150" viewBox="0 0 260 150">
                    {/* 첫 번째 게이지 (바깥쪽) */}
                    {scamTypes[0] && (
                        <>
                            {/* 배경 */}
                            <path
                                d={`M 50,130 A ${radius1},${radius1} 0 0,1 210,130`}
                                fill="none"
                                stroke="#2a2d33"
                                strokeWidth="24"
                            />
                            {/* 진행 */}
                            <path
                                d={`M 50,130 A ${radius1},${radius1} 0 0,1 210,130`}
                                fill="none"
                                stroke={getColorByPercentage(scamTypes[0].type, scamTypes[0].percentage)}
                                strokeWidth="24"
                                strokeDasharray={circumference1}
                                strokeDashoffset={getOffset(scamTypes[0].percentage, circumference1)}
                            />
                        </>
                    )}

                    {/* 두 번째 게이지 (안쪽) */}
                    {scamTypes[1] && (
                        <>
                            {/* 배경 */}
                            <path
                                d={`M 75,130 A ${radius2},${radius2} 0 0,1 185,130`}
                                fill="none"
                                stroke="#2a2d33"
                                strokeWidth="22"
                            />
                            {/* 진행 */}
                            <path
                                d={`M 75,130 A ${radius2},${radius2} 0 0,1 185,130`}
                                fill="none"
                                stroke={getColorByPercentage(scamTypes[1].type, scamTypes[1].percentage)}
                                strokeWidth="22"
                                strokeDasharray={circumference2}
                                strokeDashoffset={getOffset(scamTypes[1].percentage, circumference2)}
                            />
                        </>
                    )}
                </svg>
            </div>

            {/* 범례 */}
            <div className="legend-container">
                {scamTypes.map((scam, index) => (
                    <div key={index} className="legend-item">
                        <div 
                            className="legend-color-box" 
                            style={{ backgroundColor: getColorByPercentage(scam.type, scam.percentage) }}
                        />
                        <span className="legend-type">{scam.type}</span>
                        <span className="legend-percentage">
                            {typeof scam.percentage === 'number'
                                ? scam.percentage.toFixed(2)
                                : scam.percentage}
                            %
                        </span>
                    </div>
                ))}
            </div>

            {/* Scam Type 정보 */}
            <div className="scam-types-section">
                {token.scamTypes.map((item, index) => (
                    <div className="scam-row" key={index}>
                        <span className="scam-name">{item.type}</span>
                        <span 
                            className="scam-level"
                            style={{ color: getLevelColor(item.level) }}
                        >
                            {item.level}
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default RiskScoreCard;