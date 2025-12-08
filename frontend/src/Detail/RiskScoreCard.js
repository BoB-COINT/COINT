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
        const honeypot = scamTypes.find(s => s.type === 'Honeypot')?.percentage || 0;
        const rugPull = scamTypes.find(s => s.type === 'Rug Pull')?.percentage || 0;

        if (honeypot >= 1 && rugPull >= 1) {
            return { level: 'Critical', color: '#FF4444' };
        } else if (honeypot === 0 && rugPull >= 1) {
            return { level: 'Warning', color: '#FF9500' };
        } else if (honeypot >= 1 && rugPull === 0) {
            return { level: 'Caution', color: '#FFC107' };
        } else {
            return { level: 'Not Detected', color: '#00C853' };
        }
    };

    // 퍼센티지에 따른 색상 결정 함수
    const getColorByPercentage = (percentage) => {
        if (percentage <= 20) {
            return '#00c853'; // 초록색
        } else if (percentage <= 60) {
            return '#FFC107'; // 주황같은 노랑색
        } else {
            return '#FF4444'; // 빨간색
        }
    };

    const getOffset = (percentage, circumference) => {
        return circumference - (percentage / 100) * circumference;
    };

    const getLevelColor = (level) => {
        switch(level) {
            case 'Critical': return '#FF4444';
            case 'Warning': return '#FF9500';
            case 'Caution': return '#FFC107';
            case 'Normal': return '#00C853';
            default: return '#FFFFFF';
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
                                stroke={getColorByPercentage(scamTypes[0].percentage)}
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
                                stroke={getColorByPercentage(scamTypes[1].percentage)}
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
                            style={{ backgroundColor: getColorByPercentage(scam.percentage) }}
                        />
                        <span className="legend-type">{scam.type}</span>
                        <span className="legend-percentage">{scam.percentage}%</span>
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