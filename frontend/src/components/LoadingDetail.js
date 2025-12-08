import React, { useEffect, useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import './LoadingDetail.css';

function LoadingDetail() {
    const [searchParams] = useSearchParams();
    const tokenAddr = (searchParams.get('address') || '').trim();

    const [progress, setProgress] = useState(10);
    const navigate = useNavigate();

    const safeProgress = Math.min(Math.max(progress, 0), 100);

    useEffect(() => {
        if (!tokenAddr) return;

        // ✅ 1) 가짜 progress 타이머: 최대 90%까지만 서서히 증가
        const timer = setInterval(() => {
            setProgress((prev) => (prev < 90 ? prev + 5 : prev));
        }, 500);

        // ✅ 2) 실제 /api/analyze 호출
        const run = async () => {
            try {
                await fetch('/api/analyze/', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ token_addr: tokenAddr }),
                });

                // 완료 시 100%로
                setProgress(100);

                // 살짝 딜레이 후 상세 페이지로 이동
                setTimeout(() => {
                    navigate(`/detail?address=${tokenAddr}`);
                }, 400);
            } catch (err) {
                console.error('Analyze failed', err);
                // TODO: 여기서 에러 메시지 띄우거나 에러 페이지로 보내도 됨
            } finally {
                clearInterval(timer);
            }
        };

        run();

        // 언마운트 시 타이머 정리
        return () => clearInterval(timer);
    }, [tokenAddr, navigate]);

    return (
        <div className="loading-detail-container">
            <div className="loading-detail-content">
                <div className="loading-spinner">
                    <div className="spinner-ring"></div>
                    <div className="spinner-ring"></div>
                    <div className="spinner-ring"></div>
                </div>

                <h2 className="loading-detail-title">
                    {tokenAddr
                        ? `${tokenAddr} 토큰을 분석하고 있습니다...`
                        : '토큰 정보를 분석하고 있습니다...'}
                </h2>

                <p className="loading-detail-description">
                    잠시만 기다려주세요
                </p>

                <div className="loading-progress">
                    <div
                        className="progress-bar"
                        style={{ width: `${safeProgress}%` }}
                    />
                </div>
            </div>
        </div>
    );
}

export default LoadingDetail;
