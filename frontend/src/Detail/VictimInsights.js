import './VictimInsights.css';

export default function VictimInsightsCard({ items = [] }) {
    if (!items.length) {
        return <div style={{color: '#87888c'}}>탐지 지표가 없습니다.</div>
    }

    // 카테고리별로 데이터 분류
    const codeAnalyze = items.filter(item => item.category === 'code_analyze');
    const honeypotPattern = items.filter(item => item.category === 'honeypot');
    const rugpullPattern = items.filter(item => item.category === 'rugpull');

    const CategoryCard = ({ title, items }) => {
        if (!items.length) return null;
        
        return (
            <div className="category-card">
                <h4 className="category-title">{title}</h4>
                <ul className="category-list">
                    {items.map((item, idx) => (
                        <li key={idx} className="category-item">
                            {item.description}
                        </li>
                    ))}
                </ul>
            </div>
        );
    };

    return (
        <div>
            <div className="victim-header">
                <h3>Visitor Insights</h3>
                <span className="victim-count">탐지 지표: {items.length}개</span>
            </div>
            
            <CategoryCard 
                title="Code Analyze" 
                items={codeAnalyze}
            />
            
            <CategoryCard 
                title="Honeypot Pattern" 
                items={honeypotPattern}
            />
            
            <CategoryCard 
                title="RugPull Pattern" 
                items={rugpullPattern}
            />
        </div>
    );
}