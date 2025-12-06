"""
Unified data collector for Ethereum token analysis.
Collects token info, pair events, and holder information.
Based on the original working implementation.
"""
import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timezone

from web3 import Web3
import requests

logger = logging.getLogger(__name__)

# Event signatures
EVENT_SIGNATURES = {
    '0x4c209b5fc8ad50758f13e2e1088ba56a560dff690a1c6fef26394f4c03821c4f': 'Mint',
    '0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822': 'Swap',
    '0xdccd412f0b1252819cb1fd330b93224ca42612892bb3f4f789976e6d81936496': 'Burn',
    '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef': 'Transfer',
    '0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1': 'Sync'
}

# Factory addresses
FACTORY_LIST = {
    '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f': 'UniswapV2',
    '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac': 'SushiswapV2',
    '0x1097053Fd2ea711dad45caCcc45EfF7548fCB362': 'PancakeV2',
    '0x115934131916C8b277DD010Ee02de363c09d037c': 'ShibaSwapV1',
}

# ERC20 ABI
ERC20_ABI = [
    {"constant": True, "inputs": [], "name": "name", "outputs": [{"name": "", "type": "string"}], "type": "function"},
    {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "string"}], "type": "function"},
    {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function"},
    {"constant": True, "inputs": [], "name": "totalSupply", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
]


class UnifiedDataCollector:
    """Collects token information, pair events, and holders from Ethereum blockchain."""

    def __init__(self, rpc_url: str, etherscan_api_key: str, etherscan_api_url: str, moralis_api_key: str, chainbase_api_key: str):
        """
        Initialize the collector.

        Args:
            rpc_url: Ethereum RPC endpoint URL
            etherscan_api_key: Etherscan API key
            etherscan_api_url: Etherscan API URL
            moralis_api_key: Moralis API key
        """
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self.etherscan_api_key = etherscan_api_key
        self.etherscan_api_url = etherscan_api_url
        self.moralis_api_key = moralis_api_key
        self.chainbase_api_key = chainbase_api_key

        if not self.web3.is_connected():
            raise ConnectionError("Failed to connect to Ethereum node")

        logger.info(f"Connected to Ethereum node (chain_id={self.web3.eth.chain_id})")

    def collect_all(self, token_addr: str, days: int = None) -> Dict[str, Any]:
        """
        Collect all data for a given token address.

        Args:
            token_addr: Token contract address
            days: Number of days to collect events from pair creation (None = no limit)

        Returns:
            Dictionary containing token_info, pair_events, holders
        """
        token_addr = Web3.to_checksum_address(token_addr)
        logger.info(f"Collecting data for token: {token_addr}")

        # Find pair and collect initial events
        pair_info, initial_events, init_seen = self._find_pair_and_initial_events(token_addr)

        if not pair_info:
            logger.warning(f"No pair found for token {token_addr}")
            token_create_ts, token_creator = self._get_contract_creation_info(token_addr)
            symbol, name = self._get_token_metadata(token_addr)

            token_info = {
                'token_addr': token_addr,
                'pair_addr': '0x0000000000000000000000000000000000000000',
                'token_create_ts': token_create_ts,
                'lp_create_ts': token_create_ts,
                'pair_idx': 0,
                'pair_type': 'None',
                'token_creator_addr': token_creator,
                'symbol': symbol,
                'name': name
            }
            return {
                'token_info': token_info,
                'pair_events': [],
                'holders': []
            }

        # Collect remaining pair events (excluding initial events)
        if days is not None:
            end_block = self._get_block_after_days(
                pair_info['pair_created_block'],
                pair_info['pair_created_ts'],
                days
            )
        else:
            end_block = 'latest'

        remaining_events = self._collect_pair_events(
            pair_info['pair_addr'],
            pair_info['pair_created_block'],
            end_block,
            pair_info['lp_total_supply'],
            init_seen
        )

        # Combine initial + remaining events
        all_events = initial_events + remaining_events

        # Build token_info
        token_create_ts, token_creator = self._get_contract_creation_info(token_addr)
        symbol, name = self._get_token_metadata(token_addr)
        holder_cnt = self._get_holder_count(token_addr)

        token_info = {
            'token_addr': token_addr,
            'pair_addr': pair_info['pair_addr'],
            'pair_creator' : pair_info['pair_creator'],
            'token_create_ts': token_create_ts,
            'lp_create_ts': pair_info['lp_create_ts'],
            'pair_idx': pair_info['pair_idx'],
            'pair_type': pair_info['pair_type'],
            'token_creator_addr': token_creator,
            'symbol': symbol,
            'name': name,
            'holder_cnt': holder_cnt
        }

        # Collect holders
        holders = self._collect_holders(token_addr)

        return {
            'token_info': token_info,
            'pair_events': all_events,
            'holders': holders
        }

    def _find_pair_and_initial_events(self, token_addr: str) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Set[Tuple[str, int]]]:
        """
        Find pair from Factory PairCreated events and extract initial events from creation tx receipt.
        Returns: (pair_info, initial_events, init_seen)
        """
        logger.info(f"Searching for pair in Factory events: {token_addr}")

        token_addr_enc = '0x' + '0' * 24 + token_addr[2:].lower()

        for factory_addr, factory_name in FACTORY_LIST.items():
            for token_position in [1, 2]:
                params = {
                    'chainid': 1,
                    'module': 'logs',
                    'action': 'getLogs',
                    'fromBlock': '0',
                    'toBlock': 'latest',
                    'address': factory_addr,
                    'topic0': '0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cddefde31afa28d0e9',
                    f'topic{token_position}': token_addr_enc,
                    'apikey': self.etherscan_api_key
                }

                try:
                    response = requests.get(self.etherscan_api_url, params=params, timeout=30)
                    response.raise_for_status()
                    data = response.json()

                    if data.get('status') == '1' and data.get('result'):
                        log = data['result'][0]
                        pair_addr = '0x' + log['data'][26:66]
                        pair_created_block = int(log['blockNumber'], 16)
                        pair_created_ts = int(log['timeStamp'], 16)
                        tx_hash = log['transactionHash']
                        tx_from, tx_to = self._get_tx_from_to(tx_hash)

                        topics = log['topics']
                        token0 = Web3.to_checksum_address('0x' + topics[1][-40:])
                        token1 = Web3.to_checksum_address('0x' + topics[2][-40:])

                        pair_idx = 0 if token0.lower() == token_addr.lower() else 1

                        # Get decimals
                        token0_decimals = self._get_decimals(token0)
                        token1_decimals = self._get_decimals(token1)

                        pair_info = {
                            'pair_addr': pair_addr,
                            'pair_created_block': pair_created_block,
                            'pair_created_ts': pair_created_ts,
                            'lp_create_ts': datetime.fromtimestamp(pair_created_ts, tz=timezone.utc),
                            'pair_idx': pair_idx,
                            'pair_type': factory_name,
                            'pair_creator': tx_from,
                            'token0': token0,
                            'token1': token1,
                            'token0_decimals': token0_decimals,
                            'token1_decimals': token1_decimals,
                            'lp_total_supply': 0
                        }
                        evt_log = {
                            'token0':token0,
                            'token1':token1,
                            'pairaddr':pair_addr
                        }
                        pair_created = {
                            'timestamp': datetime.fromtimestamp(pair_created_ts, tz=timezone.utc),
                            'block_number': pair_created_block,
                            'tx_hash': tx_hash,
                            'tx_from': tx_from,
                            'tx_to': tx_to,
                            'evt_idx': int(log['logIndex'],16),
                            'evt_type': 'PairCreated',
                            'evt_log': evt_log,
                            'lp_total_supply': 0
                        }
                        events = [pair_created]
                        logger.info(f"Found pair {pair_addr} in {factory_name}")

                        # Extract initial events from creation transaction receipt
                        initial_events, lp_total_supply, init_seen = self._extract_initial_events_from_tx(
                            tx_hash, pair_info
                        )
                        pair_info['lp_total_supply'] = lp_total_supply
                        total_evt = events + initial_events
                        return pair_info, total_evt, init_seen

                except Exception as e:
                    logger.error(f"Error searching factory {factory_name}: {e}")
                    continue

        return None, [], set()

    def _extract_initial_events_from_tx(self, tx_hash: str, pair_info: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], float, Set[Tuple[str, int]]]:
        """
        Extract initial events from pair creation transaction receipt.
        Returns: (events, lp_total_supply, init_seen)
        """
        try:
            receipt = self.web3.eth.get_transaction_receipt(tx_hash)
            events = []
            init_seen = set()
            lp_total_supply = 0
            reserve = [0.0, 0.0]

            for log in receipt['logs']:
                if log['address'].lower() != pair_info['pair_addr'].lower():
                    continue

                sig = '0x' + bytes.hex(log['topics'][0])
                if sig not in EVENT_SIGNATURES:
                    continue

                evt_type = EVENT_SIGNATURES[sig]
                log_index = int(log['logIndex'])

                event = None
                if evt_type == 'Mint':
                    event, lp_total_supply = self._parse_mint_rpc(log, pair_info, reserve, lp_total_supply)
                elif evt_type == 'Burn':
                    event, lp_total_supply = self._parse_burn_rpc(log, pair_info, reserve, lp_total_supply)
                elif evt_type == 'Swap':
                    event, lp_total_supply = self._parse_swap_rpc(log, pair_info, reserve, lp_total_supply)
                elif evt_type == 'Sync':
                    event, lp_total_supply = self._parse_sync_rpc(log, pair_info, reserve, lp_total_supply)
                elif evt_type == 'Transfer':
                    topic1 = int.from_bytes(log['topics'][1], 'big')
                    topic2 = int.from_bytes(log['topics'][2], 'big')
                    if (topic1 == 0 and topic2 != 0) or (topic1 != 0 and topic2 == 0):
                        event, lp_total_supply = self._parse_transfer_rpc(log, pair_info, reserve, lp_total_supply)

                if event:
                    events.append(event)
                    init_seen.add((tx_hash, log_index))

            logger.info(f"Extracted {len(events)} initial events from creation tx")
            return events, lp_total_supply, init_seen

        except Exception as e:
            logger.error(f"Error extracting initial events: {e}")
            return [], 0.0, set()

    def _collect_pair_events(self, pair_addr: str, start_block: int, end_block, lp_total_supply, init_seen: Set[Tuple[str, int]]) -> List[Dict[str, Any]]:
        """Collect pair events using Etherscan getLogs API, excluding init_seen."""
        logger.info(f"Collecting pair events for {pair_addr} from block {start_block} to {end_block}")

        events = []
        page = 1
        max_events = 3000
        reserve = [0.0, 0.0]

        while len(events) < max_events:
            params = {
                'chainid': 1,
                'module': 'logs',
                'action': 'getLogs',
                'fromBlock': start_block,
                'toBlock': end_block,
                'address': pair_addr,
                'page': page,
                'offset': 1000,
                'apikey': self.etherscan_api_key
            }

            try:
                response = requests.get(self.etherscan_api_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if data.get('status') != '1':
                    if data.get('message') == 'No records found':
                        break
                    logger.warning(f"API error: {data.get('message', 'Unknown')}")
                    break

                logs = data.get('result', [])
                if not logs:
                    break

                for log in logs:
                    tx_hash = log['transactionHash']
                    log_index = int(log['logIndex'], 16) if log['logIndex'] != '0x' else 0

                    # Skip if already processed in initial events
                    if (tx_hash, log_index) in init_seen:
                        continue

                    event = self._parse_event_log(log, pair_addr, reserve, lp_total_supply)
                    if event:
                        events.append(event)
                        lp_total_supply = event['_lp_total_supply']

                    if len(events) >= max_events:
                        break

                if len(logs) < 1000 or len(events) >= max_events:
                    break

                page += 1
                time.sleep(0.2)

            except Exception as e:
                logger.error(f"Error collecting events (page {page}): {e}")
                break

        logger.info(f"Total events collected: {len(events)}")
        return events

    def _parse_event_log(self, log: Dict, pair_addr: str, reserve: List[float], lp_total_supply: float) -> Optional[Dict[str, Any]]:
        """Parse event log from getLogs API (non-receipt format)."""
        topics = log.get('topics', [])
        if not topics:
            return None

        event_sig = topics[0]
        if event_sig not in EVENT_SIGNATURES:
            return None

        evt_type = EVENT_SIGNATURES[event_sig]

        # Skip Transfer events that are not mint/burn
        if evt_type == 'Transfer':
            topic1 = int(topics[1], 16) if len(topics) > 1 else 1
            topic2 = int(topics[2], 16) if len(topics) > 2 else 1
            if not ((topic1 == 0 and topic2 != 0) or (topic1 != 0 and topic2 == 0)):
                return None

        # Build pair_info for parsing functions
        pair_info = {
            'pair_addr': pair_addr,
            'token0_decimals': 18,
            'token1_decimals': 18
        }

        event = None
        if evt_type == 'Mint':
            event, lp_total_supply = self._parse_mint_api(log, pair_info, reserve, lp_total_supply)
        elif evt_type == 'Burn':
            event, lp_total_supply = self._parse_burn_api(log, pair_info, reserve, lp_total_supply)
        elif evt_type == 'Swap':
            event, lp_total_supply = self._parse_swap_api(log, pair_info, reserve, lp_total_supply)
        elif evt_type == 'Sync':
            event, lp_total_supply = self._parse_sync_api(log, pair_info, reserve, lp_total_supply)
        elif evt_type == 'Transfer':
            event, lp_total_supply = self._parse_transfer_api(log, pair_info, reserve, lp_total_supply)

        if event:
            event['_lp_total_supply'] = lp_total_supply

        return event

    # Parsing functions for RPC receipt format
    def _parse_mint_rpc(self, log: Dict, pair_info: Dict, reserve: List[float], lp_total_supply) -> Tuple[Dict[str, Any], float]:
        evt_log = {}
        topics = log['topics']
        data_hex = bytes.hex(log['data'])

        if len(topics) > 1:
            evt_log['sender'] = '0x' + bytes.hex(topics[1])[-40:]

        token0_decimals = pair_info['token0_decimals']
        token1_decimals = pair_info['token1_decimals']

        if len(data_hex) >= 128:
            amount0 = int('0x' + data_hex[0:64], 16) / (10 ** token0_decimals)
            amount1 = int('0x' + data_hex[64:128], 16) / (10 ** token1_decimals)
            evt_log['amount0'] = amount0
            evt_log['amount1'] = amount1

        timestamp = datetime.fromtimestamp(int(log['blockTimestamp'], 16), tz=timezone.utc)
        tx_hash = '0x' + bytes.hex(log['transactionHash'])
        tx_from, tx_to = self._get_tx_from_to(tx_hash)

        return {
            'timestamp': timestamp,
            'block_number': int(log['blockNumber']),
            'tx_hash': tx_hash,
            'tx_from': tx_from,
            'tx_to': tx_to,
            'evt_idx': int(log['logIndex']),
            'evt_type': 'Mint',
            'evt_log': evt_log,
            'lp_total_supply': lp_total_supply
        }, lp_total_supply

    def _parse_burn_rpc(self, log: Dict, pair_info: Dict, reserve: List[float], lp_total_supply) -> Tuple[Dict[str, Any], float]:
        evt_log = {}
        topics = log['topics']
        data_hex = bytes.hex(log['data'])

        if len(topics) > 1:
            evt_log['sender'] = '0x' + bytes.hex(topics[1])[-40:]
        if len(topics) > 2:
            evt_log['to'] = '0x' + bytes.hex(topics[2])[-40:]

        token0_decimals = pair_info['token0_decimals']
        token1_decimals = pair_info['token1_decimals']

        if len(data_hex) >= 128:
            amount0 = int('0x' + data_hex[0:64], 16) / (10 ** token0_decimals)
            amount1 = int('0x' + data_hex[64:128], 16) / (10 ** token1_decimals)
            evt_log['amount0'] = amount0
            evt_log['amount1'] = amount1

        timestamp = datetime.fromtimestamp(int(log['blockTimestamp'], 16), tz=timezone.utc)
        tx_hash = '0x' + bytes.hex(log['transactionHash'])
        tx_from, tx_to = self._get_tx_from_to(tx_hash)

        return {
            'timestamp': timestamp,
            'block_number': int(log['blockNumber']),
            'tx_hash': tx_hash,
            'tx_from': tx_from,
            'tx_to': tx_to,
            'evt_idx': int(log['logIndex']),
            'evt_type': 'Burn',
            'evt_log': evt_log,
            'lp_total_supply': lp_total_supply
        }, lp_total_supply

    def _parse_swap_rpc(self, log: Dict, pair_info: Dict, reserve: List[float], lp_total_supply) -> Tuple[Dict[str, Any], float]:
        evt_log = {}
        topics = log['topics']
        data_hex = bytes.hex(log['data'])

        # if len(topics) > 1:
        #     evt_log['sender'] = '0x' + bytes.hex(topics[1])[-40:]
        # if len(topics) > 2:
        #     evt_log['to'] = '0x' + bytes.hex(topics[2])[-40:]

        token0_decimals = pair_info['token0_decimals']
        token1_decimals = pair_info['token1_decimals']

        if len(data_hex) >= 256:
            evt_log['amount0In'] = int('0x' + data_hex[0:64], 16) / (10 ** token0_decimals)
            evt_log['amount1In'] = int('0x' + data_hex[64:128], 16) / (10 ** token1_decimals)
            evt_log['amount0Out'] = int('0x' + data_hex[128:192], 16) / (10 ** token0_decimals)
            evt_log['amount1Out'] = int('0x' + data_hex[192:256], 16) / (10 ** token1_decimals)

        timestamp = datetime.fromtimestamp(int(log['blockTimestamp'], 16), tz=timezone.utc)
        tx_hash = '0x' + bytes.hex(log['transactionHash'])

        return {
            'timestamp': timestamp,
            'block_number': int(log['blockNumber']),
            'tx_hash': tx_hash,
            'tx_from': '0x0',
            'tx_to': '0x0',
            'evt_idx': int(log['logIndex']),
            'evt_type': 'Swap',
            'evt_log': evt_log,
            'lp_total_supply':lp_total_supply
        }, lp_total_supply

    def _parse_sync_rpc(self, log: Dict, pair_info: Dict, reserve: List[float], lp_total_supply) -> Tuple[Dict[str, Any], float]:
        evt_log = {}
        data_hex = bytes.hex(log['data'])

        token0_decimals = pair_info['token0_decimals']
        token1_decimals = pair_info['token1_decimals']

        if len(data_hex) >= 128:
            reserve[0] = int('0x' + data_hex[0:64], 16) / (10 ** token0_decimals)
            reserve[1] = int('0x' + data_hex[64:128], 16) / (10 ** token1_decimals)
            evt_log['reserve0'] = reserve[0]
            evt_log['reserve1'] = reserve[1]

        timestamp = datetime.fromtimestamp(int(log['blockTimestamp'], 16), tz=timezone.utc)
        tx_hash = '0x' + bytes.hex(log['transactionHash'])
        tx_from, tx_to = self._get_tx_from_to(tx_hash)

        return {
            'timestamp': timestamp,
            'block_number': int(log['blockNumber']),
            'tx_hash': tx_hash,
            'tx_from': tx_from,
            'tx_to': tx_to,
            'evt_idx': int(log['logIndex']),
            'evt_type': 'Sync',
            'evt_log': evt_log,
            'lp_total_supply': lp_total_supply
        }, lp_total_supply

    def _parse_transfer_rpc(self, log: Dict, pair_info: Dict, reserve: List[float], lp_total_supply) -> Tuple[Dict[str, Any], float]:
        evt_log = {}
        topics = log['topics']
        data_hex = '0x' + bytes.hex(log['data'])

        value = int(data_hex, 16) / 1e18

        from_addr = '0x' + bytes.hex(topics[1])[-40:]
        to_addr = '0x' + bytes.hex(topics[2])[-40:]

        evt_log['from'] = from_addr
        evt_log['to'] = to_addr
        evt_log['value'] = value

        if int.from_bytes(topics[1], 'big') == 0:
            lp_total_supply += value
        elif int.from_bytes(topics[2], 'big') == 0:
            lp_total_supply -= value

        timestamp = datetime.fromtimestamp(int(log['blockTimestamp'], 16), tz=timezone.utc)
        tx_hash = '0x' + bytes.hex(log['transactionHash'])
        tx_from, tx_to = self._get_tx_from_to(tx_hash)

        return {
            'timestamp': timestamp,
            'block_number': int(log['blockNumber']),
            'tx_hash': tx_hash,
            'tx_from': tx_from,
            'tx_to': tx_to,
            'evt_idx': int(log['logIndex']),
            'evt_type': 'Transfer',
            'evt_log': evt_log,
            'lp_total_supply': lp_total_supply
        }, lp_total_supply

    # Parsing functions for getLogs API format
    def _parse_mint_api(self, log: Dict, pair_info: Dict, reserve: List[float], lp_total_supply) -> Tuple[Dict[str, Any], float]:
        evt_log = {}
        topics = log['topics']
        data_hex = log['data'][2:]

        if len(topics) > 1:
            evt_log['sender'] = '0x' + topics[1][-40:]

        token0_decimals = pair_info['token0_decimals']
        token1_decimals = pair_info['token1_decimals']

        if len(data_hex) >= 128:
            amount0 = int('0x' + data_hex[0:64], 16) / (10 ** token0_decimals)
            amount1 = int('0x' + data_hex[64:128], 16) / (10 ** token1_decimals)
            evt_log['amount0'] = amount0
            evt_log['amount1'] = amount1

        timestamp = datetime.fromtimestamp(int(log['timeStamp'], 16), tz=timezone.utc)
        tx_hash = log['transactionHash']
        tx_from, tx_to = self._get_tx_from_to(tx_hash)

        return {
            'timestamp': timestamp,
            'block_number': int(log['blockNumber'], 16),
            'tx_hash': tx_hash,
            'tx_from': tx_from,
            'tx_to': tx_to,
            'evt_idx': int(log['logIndex'], 16) if log['logIndex'] != '0x' else 0,
            'evt_type': 'Mint',
            'evt_log': evt_log,
            'lp_total_supply': lp_total_supply
        }, lp_total_supply

    def _parse_burn_api(self, log: Dict, pair_info: Dict, reserve: List[float], lp_total_supply) -> Tuple[Dict[str, Any], float]:
        evt_log = {}
        topics = log['topics']
        data_hex = log['data'][2:]

        if len(topics) > 1:
            evt_log['sender'] = '0x' + topics[1][-40:]
        if len(topics) > 2:
            evt_log['to'] = '0x' + topics[2][-40:]

        token0_decimals = pair_info['token0_decimals']
        token1_decimals = pair_info['token1_decimals']

        if len(data_hex) >= 128:
            amount0 = int('0x' + data_hex[0:64], 16) / (10 ** token0_decimals)
            amount1 = int('0x' + data_hex[64:128], 16) / (10 ** token1_decimals)
            evt_log['amount0'] = amount0
            evt_log['amount1'] = amount1

        timestamp = datetime.fromtimestamp(int(log['timeStamp'], 16), tz=timezone.utc)
        tx_hash = log['transactionHash']
        tx_from, tx_to = self._get_tx_from_to(tx_hash)

        return {
            'timestamp': timestamp,
            'block_number': int(log['blockNumber'], 16),
            'tx_hash': tx_hash,
            'tx_from': tx_from,
            'tx_to': tx_to,
            'evt_idx': int(log['logIndex'], 16) if log['logIndex'] != '0x' else 0,
            'evt_type': 'Burn',
            'evt_log': evt_log,
            'lp_total_supply': lp_total_supply
        }, lp_total_supply

    def _parse_swap_api(self, log: Dict, pair_info: Dict, reserve: List[float], lp_total_supply) -> Tuple[Dict[str, Any], float]:
        evt_log = {}
        topics = log['topics']
        data_hex = log['data'][2:]

        if len(topics) > 1:
            evt_log['sender'] = '0x' + topics[1][-40:]
        if len(topics) > 2:
            evt_log['to'] = '0x' + topics[2][-40:]

        token0_decimals = pair_info['token0_decimals']
        token1_decimals = pair_info['token1_decimals']

        if len(data_hex) >= 256:
            evt_log['amount0In'] = int('0x' + data_hex[0:64], 16) / (10 ** token0_decimals)
            evt_log['amount1In'] = int('0x' + data_hex[64:128], 16) / (10 ** token1_decimals)
            evt_log['amount0Out'] = int('0x' + data_hex[128:192], 16) / (10 ** token0_decimals)
            evt_log['amount1Out'] = int('0x' + data_hex[192:256], 16) / (10 ** token1_decimals)

        timestamp = datetime.fromtimestamp(int(log['timeStamp'], 16), tz=timezone.utc)
        tx_hash = log['transactionHash']
        tx_from, tx_to = self._get_tx_from_to(tx_hash)

        return {
            'timestamp': timestamp,
            'block_number': int(log['blockNumber'], 16),
            'tx_hash': tx_hash,
            'tx_from': tx_from,
            'tx_to': tx_to,
            'evt_idx': int(log['logIndex'], 16) if log['logIndex'] != '0x' else 0,
            'evt_type': 'Swap',
            'evt_log': evt_log,
            'lp_total_supply': lp_total_supply
        }, lp_total_supply

    def _parse_sync_api(self, log: Dict, pair_info: Dict, reserve: List[float], lp_total_supply) -> Tuple[Dict[str, Any], float]:
        evt_log = {}
        data_hex = log['data'][2:]

        token0_decimals = pair_info['token0_decimals']
        token1_decimals = pair_info['token1_decimals']

        if len(data_hex) >= 128:
            reserve[0] = int('0x' + data_hex[0:64], 16) / (10 ** token0_decimals)
            reserve[1] = int('0x' + data_hex[64:128], 16) / (10 ** token1_decimals)
            evt_log['reserve0'] = reserve[0]
            evt_log['reserve1'] = reserve[1]

        timestamp = datetime.fromtimestamp(int(log['timeStamp'], 16), tz=timezone.utc)
        tx_hash = log['transactionHash']
        tx_from, tx_to = self._get_tx_from_to(tx_hash)

        return {
            'timestamp': timestamp,
            'block_number': int(log['blockNumber'], 16),
            'tx_hash': tx_hash,
            'tx_from': tx_from,
            'tx_to': tx_to,
            'evt_idx': int(log['logIndex'], 16) if log['logIndex'] != '0x' else 0,
            'evt_type': 'Sync',
            'evt_log': evt_log,
            'lp_total_supply': lp_total_supply
        }, lp_total_supply

    def _parse_transfer_api(self, log: Dict, pair_info: Dict, reserve: List[float], lp_total_supply) -> Tuple[Dict[str, Any], float]:
        evt_log = {}
        topics = log['topics']
        data_hex = log['data']

        value = int(data_hex, 16) / 10**18

        from_addr = '0x' + topics[1][-40:]
        to_addr = '0x' + topics[2][-40:]

        evt_log['from'] = from_addr
        evt_log['to'] = to_addr
        evt_log['value'] = value

        if int(topics[1], 16) == 0:
            lp_total_supply += value
        elif int(topics[2], 16) == 0:
            lp_total_supply -= value

        timestamp = datetime.fromtimestamp(int(log['timeStamp'], 16), tz=timezone.utc)
        tx_hash = log['transactionHash']
        tx_from, tx_to = self._get_tx_from_to(tx_hash)

        return {
            'timestamp': timestamp,
            'block_number': int(log['blockNumber'], 16),
            'tx_hash': tx_hash,
            'tx_from': tx_from,
            'tx_to': tx_to,
            'evt_idx': int(log['logIndex'], 16) if log['logIndex'] != '0x' else 0,
            'evt_type': 'Transfer',
            'evt_log': evt_log,
            'lp_total_supply': lp_total_supply
        }, lp_total_supply

    def _get_tx_from_to(self, tx_hash: str) -> tuple:
        """Get transaction from/to addresses."""
        try:
            tx = self.web3.eth.get_transaction(tx_hash)
            return (tx['from'] if tx['from'] else '0x0', tx['to'] if tx['to'] else '0x0')
        except Exception:
            return ('0x0', '0x0')

    def _get_decimals(self, token_addr: str) -> int:
        """Get token decimals."""
        try:
            contract = self.web3.eth.contract(
                address=Web3.to_checksum_address(token_addr),
                abi=ERC20_ABI
            )
            return contract.functions.decimals().call()
        except Exception:
            return 18

    def _get_block_after_days(self, start_block: int, start_ts: int, days: int) -> int:
        """Calculate block number after specified days using binary search."""
        target_time = start_ts + (days * 24 * 60 * 60)
        current_block = self.web3.eth.block_number

        low = start_block
        high = min(start_block + (days * 7200), current_block)

        try:
            while low < high:
                mid = (low + high) // 2
                block = self.web3.eth.get_block(mid)
                block_time = block['timestamp']

                if block_time < target_time:
                    low = mid + 1
                else:
                    high = mid

            return low
        except Exception as e:
            logger.error(f"Error calculating end block: {e}")
            return start_block + (days * 7200)

    def _get_contract_creation_info(self, contract_addr: str) -> tuple:
        """Get contract creation timestamp and creator."""
        params = {
            'chainid': 1,
            'module': 'contract',
            'action': 'getcontractcreation',
            'contractaddresses': contract_addr,
            'apikey': self.etherscan_api_key
        }

        try:
            response = requests.get(self.etherscan_api_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('status') == '1' and data.get('result'):
                result = data['result'][0]
                timestamp = int(result.get('timestamp', 0))
                creator = result.get('contractCreator', '')
                creation_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                return creation_time, creator
        except Exception as e:
            logger.error(f"Error getting contract creation info: {e}")

        return datetime.now(tz=timezone.utc), ''

    def _get_token_metadata(self, token_addr: str) -> tuple:
        """Get token symbol and name."""
        try:
            contract = self.web3.eth.contract(
                address=Web3.to_checksum_address(token_addr),
                abi=ERC20_ABI
            )
            symbol = contract.functions.symbol().call()
            name = contract.functions.name().call()
            return symbol, name
        except Exception as e:
            logger.warning(f"Failed to get token metadata: {e}")
            return None, None

    def _get_holder_count(self, token_addr: str) -> Optional[int]:
        """Get total holder count using Chainbase API."""
        logger.info(f"Fetching holder count for {token_addr}")

        url = "https://api.chainbase.online/v1/token/holders"
        params = {
            "chain_id": 1,
            "contract_address": token_addr,
            "page": 1,
            "limit": 1
        }
        headers = {
            "X-API-Key": self.chainbase_api_key
        }

        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("code") == 0 and "count" in data:
                holder_count = data["count"]
                logger.info(f"Holder count: {holder_count}")
                return holder_count
            else:
                logger.warning(f"Chainbase API error: {data.get('message', 'Unknown error')}")
                print("error")
                return None

        except Exception as e:
            logger.error(f"Error fetching holder count from Chainbase: {e}")
            print("error ex")

            return None

    def _collect_holders(self, token_addr: str, max_holders: int = 20) -> List[Dict[str, Any]]:
        """Collect top token holders using Moralis API."""
        logger.info(f"Collecting holders for {token_addr}")

        url = f"https://deep-index.moralis.io/api/v2.2/erc20/{token_addr}/owners"
        headers = {
            "accept": "application/json",
            "X-API-Key": self.moralis_api_key
        }

        holders = []
        cursor = None

        try:
            while len(holders) < max_holders:
                params = {"limit": 100}
                if cursor:
                    params["cursor"] = cursor

                response = requests.get(url, headers=headers, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                page_holders = data.get('result', [])
                if not page_holders:
                    break

                for holder in page_holders:
                    holders.append({
                        'holder_addr': holder.get('owner_address', ''),
                        'balance': holder.get('balance', '0'),
                        'rel_to_total': holder.get('percentage_relative_to_total_supply', 0.0)
                    })

                    if len(holders) >= max_holders:
                        break

                cursor = data.get('cursor')
                if not cursor:
                    break

                time.sleep(0.2)

            logger.info(f"Collected {len(holders)} holders")
            return holders

        except Exception as e:
            logger.error(f"Error fetching holders from Moralis: {e}")
            return []
