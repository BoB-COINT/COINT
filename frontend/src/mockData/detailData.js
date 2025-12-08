export const mockDetailData = {
    // Token Info
    id: 1,
    address: "0x1533C795EA2B33999Dd6eff0256640dC3b2415C2",
    tokenName: "BitDao",
    symbol: "BDO",
    tokenType: "ERC-20",
    contractOwner: "0x1533C795EA2B33999Dd6eff0256640dC3b2415C2",
    pair: "0x1533C795EA2B33999Dd6eff0256640dC3b2415C2",
    tokenCreateTs: "2021-06-17 04:23:55+00:00",
    pairCreateTs:  "2021-06-17 04:26:55+00:00",

    // Holder Info
    holders: [
        {rank: 1, address: '0x1533C795EA2B33999Dd6eff0256640dC3b2415C2', percentage: 85},
        {rank: 2, address: '0x1533C795EA2B33999Dd6eff0256640dC3b2415C2', percentage: 8},
        {rank: 3, address: '0x1533C795EA2B33999Dd6eff0256640dC3b2415C2', percentage: 1.5},
        {rank: 4, address: '0x1533C795EA2B33999Dd6eff0256640dC3b2415C2', percentage: 1},
        {rank: 5, address: '0x1533C795EA2B33999Dd6eff0256640dC3b2415C2', percentage: 0.5},
        {rank: 6, address: '0x1533C795EA2B33999Dd6eff0256640dC3b2415C2', percentage: 0.5},
        {rank: 7, address: '0x1533C795EA2B33999Dd6eff0256640dC3b2415C2', percentage: 0.5},
        {rank: 8, address: '0x1533C795EA2B33999Dd6eff0256640dC3b2415C2', percentage: 0.2},
        {rank: 9, address: '0x1533C795EA2B33999Dd6eff0256640dC3b2415C2', percentage: 0.12},
        {rank: 10, address: '0x1533C795EA2B33999Dd6eff0256640dC3b2415C2', percentage: 0.112},
        {rank: 11, address: '0x1533C795EA2B33999Dd6eff0256640dC3b2415C2', percentage: 0.02}
    ],

    // Scam Type Distribution
    scamTypeDistribution: [
        { 
            type: "Honeypot", 
            percentage: 65.3
        },
        { 
            type: "Rug Pull", 
            percentage: 23.7
        }
    ],

    // Scam Type
    scamTypes: [
        { type: "Honeypot", level: "Warning" },
        { type: "Exit", level: "Critical" }
    ],

    // Victim Insights
    victimInsights: [
        { 
            category: "code_analyze",
            title: "코드 분석: Blacklist",
            description: "일반 사용자는 매도 불가한 blacklist 로직이 존재함",
        },
        {
            category: "code_analyze",
            title: "코드 분석: 외부라우터호출",
            description: "외부 라우터를 호출하는 로직이 존재함"
        },
        {
            category: "honeypot",
            title: "허니팟 패턴",
            description: "지분 쏠림(지니계수) : 0.91 (상위 지갑에 지분이 과도하게 몰린 상태)"
        },
        {
            category: "honeypot",
            title: "허니팟 패턴",
            description: "매수-매도 로그 거래량 차(vol_log_diff) : 1.27 (양수일수록 매수 과열·매도 위축)"
        },  
        {
            category: "rugpull",
            title: "러그풀 패턴",
            description: "관찰 구간에서 리저브의 하락률(reserve_base_drop_frac) = 0.00120"
        },  
        {
            category: "rugpull",
            title: "러그풀 패턴",
            description: "해당 구간의 누적 ‘매도 스왑량’(swap_sell_to_reserve_ratio) = 0.86920"
        }
    ]
}

export default mockDetailData;