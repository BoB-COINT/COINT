import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import './Detail.css';
import TokenInfoCard from '../Detail/TokenInfoCard';
import { mockDetailData } from '../mockData/detailData';
import RiskScoreCard from '../Detail/RiskScoreCard';
import HoldersChart from '../Detail/HoldersChart';
import EmptyDetailState from '../components/EmptyDetailState';
import LoadingDetail from '../components/LoadingDetail';
import VictimInsightsCard from '../Detail/VictimInsights';

function Detail() {
    const [searchParams] = useSearchParams();
    const address = searchParams.get('address');
    
    const [tokenData, setTokenData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!address) {
            setTokenData(null);
            setLoading(false);
            return;
        }

        // 주소가 있으면 데이터 조회
        const fetchTokenData = async () => {
            setLoading(true);
            setError(null);

            try {
                // mockData에서 해당 주소가 있는지 확인
                if (mockDetailData.address.toLowerCase() === address.toLowerCase()) {
                    // 즉시 데이터 표시 (mockData에 있는 경우)
                    setTokenData(mockDetailData);
                    setLoading(false);
                } else {
                    // mockData에 없는 경우 - 로딩 중 화면 표시
                    // 실제로는 여기서 API 호출이 일어날 것임
                    setTimeout(() => {
                        // API 호출 시뮬레이션 (2초)
                        // 나중에 실제 API/DB 조회로 교체
                        setError('해당 주소의 데이터를 찾을 수 없습니다.');
                        setLoading(false);
                    }, 2000);
                }
            } catch (err) {
                setError('데이터를 불러오는 중 오류가 발생했습니다.');
                setLoading(false);
            }
        };

        fetchTokenData();
    }, [address]);

    // 주소가 없는 경우 - 안내 화면
    if (!address) {
        return <EmptyDetailState />;
    }

    // 로딩 중인 경우
    if (loading) {
        return <LoadingDetail />;
    }

    if (error) {
        return (
            <div style={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                minHeight: 'calc(100vh - 200px)',
                color: '#87888C',
                fontSize: '18px'
            }}>
                {error}
            </div>
        );
    }

    if (!tokenData) {
        return null;
    }

    return (
        <div className="detail-container">
            <div className="detail-risk-score">
                <RiskScoreCard token={tokenData} />
            </div>
            <div className="detail-token-info">
                <TokenInfoCard token={tokenData} />
            </div>
            <div className="detail-holders">
                <HoldersChart token={tokenData} />
            </div>
            <div className="detail-victim-insights">
                <VictimInsightsCard items={tokenData.victimInsights ?? []} />
            </div>
        </div>
    );
}

export default Detail;