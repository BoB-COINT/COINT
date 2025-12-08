import React from 'react';
import './LoadingDetail.css';

function LoadingDetail({progress = 60}) {

    const safeProgress = Math.min(Math.max(progress, 0), 100);

    return (
        <div className="loading-detail-container">
            <div className="loading-detail-content">
                <div className="loading-spinner">
                    <div className="spinner-ring"></div>
                    <div className="spinner-ring"></div>
                    <div className="spinner-ring"></div>
                </div>
                <h2 className="loading-detail-title">토큰 정보를 분석하고 있습니다...</h2>
                <p className="loading-detail-description">
                    잠시만 기다려주세요
                </p>
                <div className="loading-progress">
                    <div
                        className="progress-bar"
                        style={{ width: `${safeProgress}%` }}  // 🔹 여기만 변경
                    />
                </div>
            </div>
        </div>
    );
}

export default LoadingDetail;