import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import './Detail.css';
import TokenInfoCard from '../Detail/TokenInfoCard';
import RiskScoreCard from '../Detail/RiskScoreCard';
import HoldersChart from '../Detail/HoldersChart';
import EmptyDetailState from '../components/EmptyDetailState';
import LoadingDetail from '../components/LoadingDetail';
import VictimInsightsCard from '../Detail/VictimInsights';

// Result API → Detail 컴포넌트에서 쓰는 tokenData 형태로 매핑
function mapApiResultToTokenData(apiResult) {
  const snapshot = apiResult.tokenSnapshot || {};
  const holderSnapshot = apiResult.holderSnapshot || {};
  const riskScore = apiResult.riskScore || {};
  const scamTypesRaw = apiResult.scam_types || [];

  // 1) RiskScoreCard용 게이지 데이터 ------------------------------------
  const scamTypeDistribution = [];
    if (typeof riskScore.honeypot === 'number') {
    scamTypeDistribution.push({
        type: 'Honeypot',
        // ⬇ 반올림 제거: 그대로 0~100 사이의 실수 유지
        percentage: riskScore.honeypot * 100,
    });
    }
    if (typeof riskScore.exit === 'number') {
    scamTypeDistribution.push({
        type: 'Exit',
        percentage: riskScore.exit * 100,
    });
    }

  // 2) RiskScoreCard 하단 리스트용 scamTypes -----------------------------
  const scamTypes = scamTypesRaw.map((s) => ({
    type:
      s.type === 'exit'
        ? 'Exit'
        : s.type
        ? s.type.charAt(0).toUpperCase() + s.type.slice(1)
        : '-',
    level: s.level || '-',
  }));

  // 3) HoldersChart용 상위 홀더 리스트 -----------------------------------
    const holdersRaw =
    holderSnapshot.top_holders ||
    holderSnapshot.holders ||
    [];

  const holders = holdersRaw.map((h, idx) => {
    const relRaw =
      h.rel_to_total ??
      h.share_pct ??
      h.percentage ??
      0;

    let relNum =
      typeof relRaw === 'string' ? parseFloat(relRaw) : (relRaw || 0);

    // 0~100 사이로 클램핑 (바 폭, %표시용)
    if (!Number.isFinite(relNum)) relNum = 0;
    const pctForBar = Math.min(100, Math.max(0, relNum));

    return {
        rank: h.rank ?? idx + 1,
        address: h.holder_addr || h.address || '-',
        percentage: relNum,       // 표시용: 원래 값 그대로
        barPercentage: pctForBar, // 막대용: 0~100
    };
    });

  const totalHolders =
    snapshot.holder_cnt ??
    snapshot.total_holders ??
    holderSnapshot.total_holders ??
    holdersRaw.length;

  // 4) VictimInsightsCard용 인사이트 합치기 ------------------------------
  const daObj =
    apiResult.honeypotDaInsight &&
    typeof apiResult.honeypotDaInsight === 'object'
      ? apiResult.honeypotDaInsight
      : {};

  const mlArr = Array.isArray(apiResult.honeypotMlInsight)
    ? apiResult.honeypotMlInsight
    : [];

  const exitObj =
    apiResult.exitInsight &&
    typeof apiResult.exitInsight === 'object'
      ? apiResult.exitInsight
      : {};

  const victimInsights = [];

  // (1) 코드 분석 기반 인사이트 (DA) → category: code_analyze
  Object.entries(daObj).forEach(([key, val]) => {
    victimInsights.push({
      category: 'code_analyze',
      description: `${key}: ${String(val)}`,
    });
  });

  // (2) Honeypot ML 특징 → category: honeypot
  mlArr.forEach((item) => {
    victimInsights.push({
      category: 'honeypot',
      description: `${item.feat}: ${item.value}`,
    });
  });

  // (3) Exit 인사이트 → category: rugpull
  Object.entries(exitObj).forEach(([key, val]) => {
    victimInsights.push({
      category: 'rugpull',
      description: `${key}: ${String(val)}`,
    });
  });

  // 5) TokenInfoCard에서 기대하는 필드 이름으로 맞추기 -----------------
  return {
    address: apiResult.token_addr,
    tokenName: snapshot.name || snapshot.token_name || '-',
    symbol: snapshot.symbol || '-',
    tokenType: snapshot.pair_type || '-',          // UniswapV2 등
    contractOwner: snapshot.pair_creator || '-',   // owner 없어서 pair_creator 사용
    pair: snapshot.pair_addr || snapshot.pair_address || '-',
    tokenCreateTs: snapshot.token_create_ts || '',
    pairCreateTs: snapshot.lp_create_ts || snapshot.pair_create_ts || '',
    scamTypeDistribution,
    scamTypes,
    holders,
    totalHolders,
    victimInsights,
  };
}


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

        const fetchTokenData = async () => {
            setLoading(true);
            setError(null);

            try {
            const res = await fetch(`/api/results/${address}/`);

            if (!res.ok) {
                if (res.status === 404) {
                setError('해당 주소의 결과가 없습니다. 먼저 분석을 실행해 주세요.');
                } else {
                setError('서버에서 데이터를 불러오는 중 오류가 발생했습니다.');
                }
                setTokenData(null);
            } else {
                const json = await res.json();
                const mapped = mapApiResultToTokenData(json);
                setTokenData(mapped);
            }
            } catch (err) {
            console.error(err);
            setError('네트워크 오류가 발생했습니다.');
            setTokenData(null);
            } finally {
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