import './HoldersChart.css';

function HoldersChart({ token }) {

    const totalHolders = token.totalHolders ?? token.holders?.length ?? 0;

    return (
        <div className="holders-chart">
                  <h3 className="card-title">
                    Holders{totalHolders ? ` (${totalHolders} token holders)` : ''}
                </h3>
                <div className="holders-scroll">
                    {token.holders.map((holder) => (
                    <div className="holder-item" key={holder.rank}>
                        <span className="holder-address">{holder.address}</span>

                        <div className="bar-container">
                        <div
                            className="bar-fill"
                            style={{ width: `${holder.barPercentage ?? holder.percentage}%` }}
                        />
                        </div>

                        <span className="holder-percentage">
                        {Number.isFinite(holder.percentage)
                            ? `${holder.percentage.toFixed(2)}%`
                            : `${holder.percentage}%`}
                        </span>
                    </div>
                ))}  
                </div>
            
        </div>
    )
}

export default HoldersChart;