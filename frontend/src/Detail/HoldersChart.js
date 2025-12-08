import './HoldersChart.css';

function HoldersChart({ token }) {

    return (
        <div className="holders-chart">
            <h3 className="card-title">Holders</h3>
                <div className="holders-scroll">
                    {token.holders.map((holder) => (
                    <div className="holder-item" key={holder.rank}>
                        <span className="holder-address">{holder.address}</span>

                        <div className="bar-container">
                            <div
                                className="bar-fill"
                                style={{ width: `${holder.percentage}%` }}
                            />
                        </div>
                        
                        <span className="holder-percentage">{holder.percentage}%</span>
                    </div>
                ))}  
                </div>
            
        </div>
    )
}

export default HoldersChart;