"""
Step 8 (Production): Full 14-Node Heterogeneous Graph Builder
=================================================================
Builds the complete graphforANN research graph with ALL 14 node types
as per the original architecture specification.

Implemented Node Types (14):
  1.  account          — financial accounts (primary fraud target)
  2.  merchant         — payment destination entities (M-prefix)
  3.  atm              — ATM machines (ATM_XXXX receiver nodes)
  4.  device           — device fingerprints (D-prefix)
  5.  ip_address       — network origins
  6.  bank             — federated bank silos (BANK_A–BANK_J)
  7.  channel          — transaction channels (Mobile App, Web Portal, Branch, ATM)
  8.  customer         — synthetic customer owners (1:1 with account, C-prefix)
  9.  merchant_sector  — sector categories (Grocery, Electronics, Crypto_Exchange...)
  10. typology         — fraud ring typology labels (Smurfing, Layering...)
  11. ring             — fraud ring membership (RING_XXXX)
  12. case             — SAR investigation case (CASE-XXXX)
  13. country          — derived from IP block / bank jurisdiction (10 pseudo-countries)
  14. crypto_wallet    — Crypto_Exchange merchant sector → pseudo-wallet node

Implemented Edge Relations (18):
  (account, transfers_to, account)         — P2P transfer
  (account, buys_from, merchant)           — payment
  (account, withdraws_from, atm)           — ATM cash-out
  (account, uses_device, device)           — device fingerprint
  (account, from_ip, ip_address)           — network origin
  (account, belongs_to, bank)              — silo membership
  (account, transacts_via, channel)        — transaction channel
  (account, owned_by, customer)            — ownership
  (merchant, operates_in, merchant_sector) — sector classification
  (account, involved_in, typology)         — fraud typology label
  (account, member_of, ring)              — fraud ring membership
  (account, subject_of, case)             — SAR case subject
  (bank, operates_in, country)            — bank jurisdiction
  (ip_address, geolocated_in, country)    — IP country block
  (account, has_wallet, crypto_wallet)    — crypto wallet link
  (atm, located_in, bank)                 — ATM bank ownership
  (device, assigned_to, channel)          — device-channel association
  (merchant, accepts_crypto, crypto_wallet) — merchant crypto acceptance

Output:
  data/graph/hetero_graph_14node.pt

Usage:
  python src/graph/builder.py
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.data import HeteroData

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.engineer import compute_account_features


# ─── Pseudo-country assignment (by IP first octet) ─────────────────────────
def ip_to_country(ip: str) -> str:
    """Map first IP octet to a pseudo country code for graph purposes."""
    try:
        first = int(ip.split('.')[0])
    except Exception:
        return 'XX'
    brackets = [
        (0,   25,  'US'), (26,  50,  'GB'), (51,  75,  'DE'),
        (76,  100, 'IN'), (101, 125, 'SG'), (126, 150, 'JP'),
        (151, 175, 'AU'), (176, 200, 'BR'), (201, 225, 'ZA'),
        (226, 255, 'CN')
    ]
    for lo, hi, cc in brackets:
        if lo <= first <= hi:
            return cc
    return 'XX'


class FullHeteroGraphBuilder:
    """
    Production 14-Node Heterogeneous Graph Builder for graphforANN.

    Builds a publication-quality HeteroData object integrating all
    available structured signals from the 20M transaction dataset.
    """

    def __init__(self, base_dir: str = None):
        self.base_dir  = Path(base_dir) if base_dir else Path(__file__).resolve().parents[2]
        self.raw_dir   = self.base_dir / "data" / "20m_dataset"
        self.graph_dir = self.base_dir / "data" / "graph"
        self.graph_dir.mkdir(parents=True, exist_ok=True)

    # ── Data Loading ──────────────────────────────────────────────────────────

    def _load_partitions(self, max_partitions: int = 2) -> pd.DataFrame:
        part_files = sorted(self.raw_dir.glob("part_*.csv"))[:max_partitions]
        print(f"   Loading {len(part_files)} partition(s)...")
        dfs = [pd.read_csv(f) for f in part_files]
        tx = pd.concat(dfs, ignore_index=True)
        # Derive sender_bank from bank_id column
        if 'bank_id' in tx.columns and 'sender_bank' not in tx.columns:
            tx['sender_bank'] = tx['bank_id']
        print(f"   Loaded {len(tx):,} transactions | columns: {list(tx.columns)}")
        return tx

    # ── ID Mappings ────────────────────────────────────────────────────────────

    def _build_id_maps(self, tx: pd.DataFrame) -> dict:
        maps = {}

        # 1. Account (A-prefix senders)
        accs = np.array(sorted(tx[tx['sender'].str.startswith('A', na=False)]['sender'].unique()))
        maps['account']  = {a: i for i, a in enumerate(accs)}
        maps['_account_arr'] = accs

        # 2. Merchant (M-prefix receivers)
        merchs = np.array(sorted(tx[tx['receiver'].str.startswith('M', na=False)]['receiver'].unique()))
        maps['merchant']  = {m: i for i, m in enumerate(merchs)}
        maps['_merchant_arr'] = merchs

        # 3. ATM (ATM_XXXX receivers)
        atms = np.array(sorted(tx[tx['receiver'].str.startswith('ATM_', na=False)]['receiver'].unique()))
        maps['atm']  = {a: i for i, a in enumerate(atms)}
        maps['_atm_arr'] = atms

        # 4. Device
        devs = np.array(sorted(tx['device_id'].dropna().unique()))
        maps['device'] = {d: i for i, d in enumerate(devs)}
        maps['_device_arr'] = devs

        # 5. IP Address
        ips = np.array(sorted(tx['ip_address'].dropna().unique()))
        maps['ip_address'] = {ip: i for i, ip in enumerate(ips)}
        maps['_ip_arr'] = ips

        # 6. Bank
        banks = np.array(sorted(tx['bank_id'].dropna().unique()))
        maps['bank'] = {b: i for i, b in enumerate(banks)}
        maps['_bank_arr'] = banks

        # 7. Channel (transaction channel)
        channels = np.array(sorted(tx['channel'].dropna().unique()))
        maps['channel'] = {c: i for i, c in enumerate(channels)}
        maps['_channel_arr'] = channels

        # 8. Customer (synthetic — 1:1 with account, C-prefix)
        custs = np.array(['C' + a[1:] for a in accs])
        maps['customer'] = {c: i for i, c in enumerate(custs)}
        maps['_customer_arr'] = custs

        # 9. Merchant Sector
        sectors = np.array(sorted(tx['merchant_sector'].dropna().unique()))
        maps['merchant_sector'] = {s: i for i, s in enumerate(sectors)}
        maps['_sector_arr'] = sectors

        # 10. Typology (fraud pattern labels)
        typs = np.array(sorted(tx['typology'].dropna().unique()))
        maps['typology'] = {t: i for i, t in enumerate(typs)}
        maps['_typology_arr'] = typs

        # 11. Ring (fraud ring membership)
        rings = np.array(sorted(tx[tx['ring_id'] != 'NORMAL']['ring_id'].dropna().unique()))
        maps['ring'] = {r: i for i, r in enumerate(rings)}
        maps['_ring_arr'] = rings

        # 12. Case (SAR investigation case)
        cases = np.array(sorted(tx[tx['case_id'] != 'NORMAL']['case_id'].dropna().unique()))
        maps['case'] = {c: i for i, c in enumerate(cases)}
        maps['_case_arr'] = cases

        # 13. Country (derived from IP first octet)
        tx['_country'] = tx['ip_address'].apply(ip_to_country)
        countries = np.array(sorted(tx['_country'].unique()))
        maps['country'] = {c: i for i, c in enumerate(countries)}
        maps['_country_arr'] = countries

        # 14. Crypto Wallet (accounts that paid into Crypto_Exchange merchants)
        crypto_senders = tx[tx['merchant_sector'] == 'Crypto_Exchange']['sender'].dropna().unique()
        crypto_wallets = np.array(['CW' + a[1:] for a in crypto_senders if a in maps['account']])
        cw_to_acc = {cw: 'A' + cw[2:] for cw in crypto_wallets}
        maps['crypto_wallet'] = {cw: i for i, cw in enumerate(crypto_wallets)}
        maps['_crypto_wallet_arr'] = crypto_wallets
        maps['_cw_to_acc'] = cw_to_acc

        for k, v in maps.items():
            if not k.startswith('_'):
                print(f"   Node type '{k}': {len(v):,} nodes")

        return maps

    # ── Node Feature Matrices ──────────────────────────────────────────────────

    def _build_node_features(self, tx: pd.DataFrame, maps: dict) -> dict:
        feats = {}
        accs  = maps['_account_arr']

        # Account: 30-dim behavioral features (from Step 9)
        print("   Computing 30-dim account features (Step 9)...")
        feats['account'] = torch.tensor(compute_account_features(tx, accs), dtype=torch.float)

        # Merchant: 4-dim (tx_count, avg_amount, fraud_rate, sector_one_hot placeholder)
        n_m = len(maps['merchant'])
        m_stats = tx[tx['receiver'].isin(maps['merchant'])].groupby('receiver').agg(
            cnt=('amount','count'), avg=('amount','mean'), fraud=('is_fraud','mean')
        ).reset_index()
        mf = np.zeros((n_m, 4), np.float32)
        for _, r in m_stats.iterrows():
            if r['receiver'] in maps['merchant']:
                i = maps['merchant'][r['receiver']]
                mf[i, 0] = r['cnt']
                mf[i, 1] = r['avg']
                mf[i, 2] = r['fraud']
        mf[:, 3] = 1.0  # placeholder for sector embedding
        feats['merchant'] = torch.tensor(mf / (mf.max(axis=0, keepdims=True) + 1e-8), dtype=torch.float)

        # ATM: 3-dim (withdrawal_count, avg_amount, fraud_rate)
        n_atm = len(maps['atm'])
        atm_stats = tx[tx['receiver'].isin(maps['atm'])].groupby('receiver').agg(
            cnt=('amount','count'), avg=('amount','mean'), fraud=('is_fraud','mean')
        ).reset_index()
        af = np.zeros((n_atm, 3), np.float32)
        for _, r in atm_stats.iterrows():
            if r['receiver'] in maps['atm']:
                i = maps['atm'][r['receiver']]
                af[i, 0] = r['cnt']
                af[i, 1] = r['avg']
                af[i, 2] = r['fraud']
        feats['atm'] = torch.tensor(af / (af.max(axis=0, keepdims=True) + 1e-8), dtype=torch.float)

        # Device: 3-dim (usage_count, unique_accounts, fraud_rate)
        n_dev = len(maps['device'])
        dev_stats = tx[tx['device_id'].isin(maps['device'])].groupby('device_id').agg(
            cnt=('transaction_id','count'), uniq=('sender','nunique'), fraud=('is_fraud','mean')
        ).reset_index()
        df2 = np.zeros((n_dev, 3), np.float32)
        for _, r in dev_stats.iterrows():
            if r['device_id'] in maps['device']:
                i = maps['device'][r['device_id']]
                df2[i, 0] = r['cnt']
                df2[i, 1] = r['uniq']
                df2[i, 2] = r['fraud']
        feats['device'] = torch.tensor(df2 / (df2.max(axis=0, keepdims=True) + 1e-8), dtype=torch.float)

        # IP Address: 3-dim (usage_count, unique_senders, fraud_rate)
        n_ip = len(maps['ip_address'])
        ip_stats = tx[tx['ip_address'].isin(maps['ip_address'])].groupby('ip_address').agg(
            cnt=('transaction_id','count'), uniq=('sender','nunique'), fraud=('is_fraud','mean')
        ).reset_index()
        ipf = np.zeros((n_ip, 3), np.float32)
        for _, r in ip_stats.iterrows():
            if r['ip_address'] in maps['ip_address']:
                i = maps['ip_address'][r['ip_address']]
                ipf[i, 0] = r['cnt']
                ipf[i, 1] = r['uniq']
                ipf[i, 2] = r['fraud']
        feats['ip_address'] = torch.tensor(ipf / (ipf.max(axis=0, keepdims=True) + 1e-8), dtype=torch.float)

        # Bank: 2-dim (transaction_count, fraud_rate)
        n_bk = len(maps['bank'])
        bk_stats = tx[tx['bank_id'].isin(maps['bank'])].groupby('bank_id').agg(
            cnt=('transaction_id','count'), fraud=('is_fraud','mean')
        ).reset_index()
        bkf = np.zeros((n_bk, 2), np.float32)
        for _, r in bk_stats.iterrows():
            if r['bank_id'] in maps['bank']:
                i = maps['bank'][r['bank_id']]
                bkf[i, 0] = r['cnt']
                bkf[i, 1] = r['fraud']
        feats['bank'] = torch.tensor(bkf / (bkf.max(axis=0, keepdims=True) + 1e-8), dtype=torch.float)

        # Channel, Customer, Merchant_Sector, Typology, Ring, Case, Country, Crypto_Wallet
        # → one-hot / placeholder feature vectors (identity features for graph connectivity)
        for nt in ['channel', 'customer', 'merchant_sector', 'typology',
                   'ring', 'case', 'country', 'crypto_wallet']:
            n = len(maps[nt])
            feats[nt] = torch.zeros((n, 2), dtype=torch.float)  # minimal placeholder

        return feats

    # ── Label & Masks ─────────────────────────────────────────────────────────

    def _build_labels_masks(self, tx: pd.DataFrame, maps: dict):
        accs  = maps['_account_arr']
        fraud = set(tx[tx['is_fraud'] == 1]['sender'].dropna().unique())
        labels = torch.tensor([1 if a in fraud else 0 for a in accs], dtype=torch.long)
        n = len(accs)

        fraud_idx = (labels == 1).nonzero(as_tuple=True)[0].numpy()
        legit_idx = (labels == 0).nonzero(as_tuple=True)[0].numpy()
        rng = np.random.default_rng(42)
        rng.shuffle(fraud_idx)
        rng.shuffle(legit_idx)

        def split70_15_15(idx):
            n = len(idx)
            return idx[:int(.70*n)], idx[int(.70*n):int(.85*n)], idx[int(.85*n):]

        f_tr, f_val, f_te = split70_15_15(fraud_idx)
        l_tr, l_val, l_te = split70_15_15(legit_idx)
        tm = torch.zeros(n, dtype=torch.bool); tm[np.concatenate([f_tr, l_tr])]   = True
        vm = torch.zeros(n, dtype=torch.bool); vm[np.concatenate([f_val, l_val])] = True
        em = torch.zeros(n, dtype=torch.bool); em[np.concatenate([f_te, l_te])]   = True

        print(f"   Labels  -> fraud={labels.sum().item():,}/{n:,} ({labels.float().mean().item()*100:.2f}%)")
        print(f"   Masks   -> train={tm.sum():,} | val={vm.sum():,} | test={em.sum():,}")
        return labels, tm, vm, em

    # ── Edge Construction ─────────────────────────────────────────────────────

    def _make_edge(self, src_list, dst_list):
        return torch.tensor([src_list, dst_list], dtype=torch.long)

    def _build_edges(self, tx: pd.DataFrame, maps: dict) -> dict:
        acc = maps['account'];  merch = maps['merchant']
        atm = maps['atm'];      dev   = maps['device']
        ip  = maps['ip_address']; bank = maps['bank']
        chan= maps['channel'];  cust  = maps['customer']
        sec = maps['merchant_sector']; typ = maps['typology']
        ring= maps['ring'];     case_ = maps['case']
        cntry=maps['country'];  cw    = maps['crypto_wallet']
        cw2a= maps['_cw_to_acc']

        edges = {}

        def _idx(col_vals, lookup):
            mask  = col_vals.isin(lookup)
            return mask, [lookup[v] for v in col_vals[mask]]

        # ─── 1. account -> transfers_to -> account ─────────────────────────────
        p2p = tx[(tx['type']=='TRANSFER') & tx['sender'].isin(acc) & tx['receiver'].isin(acc)]
        edges[('account','transfers_to','account')] = self._make_edge(
            [acc[s] for s in p2p['sender']], [acc[r] for r in p2p['receiver']])
        print(f"   Edge account->transfers_to->account : {len(p2p):,}")

        # ─── 2. account -> buys_from -> merchant ───────────────────────────────
        pay = tx[(tx['type']=='PAYMENT') & tx['sender'].isin(acc) & tx['receiver'].isin(merch)]
        edges[('account','buys_from','merchant')] = self._make_edge(
            [acc[s] for s in pay['sender']], [merch[r] for r in pay['receiver']])
        print(f"   Edge account->buys_from->merchant   : {len(pay):,}")

        # ─── 3. account -> withdraws_from -> atm ───────────────────────────────
        atm_tx = tx[tx['sender'].isin(acc) & tx['receiver'].isin(atm)]
        edges[('account','withdraws_from','atm')] = self._make_edge(
            [acc[s] for s in atm_tx['sender']], [atm[r] for r in atm_tx['receiver']])
        print(f"   Edge account->withdraws_from->atm   : {len(atm_tx):,}")

        # ─── 4. account -> uses_device -> device ───────────────────────────────
        d_tx = tx[tx['sender'].isin(acc) & tx['device_id'].isin(dev)].drop_duplicates(['sender','device_id'])
        edges[('account','uses_device','device')] = self._make_edge(
            [acc[s] for s in d_tx['sender']], [dev[d] for d in d_tx['device_id']])
        print(f"   Edge account->uses_device->device   : {len(d_tx):,}")

        # ─── 5. account -> from_ip -> ip_address ───────────────────────────────
        ip_tx = tx[tx['sender'].isin(acc) & tx['ip_address'].isin(ip)].drop_duplicates(['sender','ip_address'])
        edges[('account','from_ip','ip_address')] = self._make_edge(
            [acc[s] for s in ip_tx['sender']], [ip[i] for i in ip_tx['ip_address']])
        print(f"   Edge account->from_ip->ip_address   : {len(ip_tx):,}")

        # ─── 6. account -> belongs_to -> bank ──────────────────────────────────
        bk_tx = tx[tx['sender'].isin(acc) & tx['bank_id'].isin(bank)].drop_duplicates(['sender','bank_id'])
        edges[('account','belongs_to','bank')] = self._make_edge(
            [acc[s] for s in bk_tx['sender']], [bank[b] for b in bk_tx['bank_id']])
        print(f"   Edge account->belongs_to->bank      : {len(bk_tx):,}")

        # ─── 7. account -> transacts_via -> channel ────────────────────────────
        ch_tx = tx[tx['sender'].isin(acc) & tx['channel'].isin(chan)].drop_duplicates(['sender','channel'])
        edges[('account','transacts_via','channel')] = self._make_edge(
            [acc[s] for s in ch_tx['sender']], [chan[c] for c in ch_tx['channel']])
        print(f"   Edge account->transacts_via->channel: {len(ch_tx):,}")

        # ─── 8. account -> owned_by -> customer ────────────────────────────────
        # Every account has exactly one synthetic customer
        accs_arr = maps['_account_arr']
        edges[('account','owned_by','customer')] = self._make_edge(
            list(range(len(accs_arr))), list(range(len(accs_arr))))  # 1-to-1 mapping
        print(f"   Edge account->owned_by->customer    : {len(accs_arr):,}")

        # ─── 9. merchant -> operates_in -> merchant_sector ─────────────────────
        m_sec = tx[tx['receiver'].isin(merch) & tx['merchant_sector'].isin(sec) &
                   (tx['merchant_sector'] != 'N/A')].drop_duplicates(['receiver','merchant_sector'])
        edges[('merchant','operates_in','merchant_sector')] = self._make_edge(
            [merch[r] for r in m_sec['receiver']], [sec[s] for s in m_sec['merchant_sector']])
        print(f"   Edge merchant->operates_in->sector  : {len(m_sec):,}")

        # ─── 10. account -> involved_in -> typology ────────────────────────────
        fraud_tx = tx[tx['sender'].isin(acc) & tx['typology'].isin(typ) &
                      (tx['typology'] != 'BENIGN')].drop_duplicates(['sender','typology'])
        edges[('account','involved_in','typology')] = self._make_edge(
            [acc[s] for s in fraud_tx['sender']], [typ[t] for t in fraud_tx['typology']])
        print(f"   Edge account->involved_in->typology : {len(fraud_tx):,}")

        # ─── 11. account -> member_of -> ring ──────────────────────────────────
        ring_tx = tx[tx['sender'].isin(acc) & tx['ring_id'].isin(ring)].drop_duplicates(['sender','ring_id'])
        edges[('account','member_of','ring')] = self._make_edge(
            [acc[s] for s in ring_tx['sender']], [ring[r] for r in ring_tx['ring_id']])
        print(f"   Edge account->member_of->ring       : {len(ring_tx):,}")

        # ─── 12. account -> subject_of -> case ─────────────────────────────────
        case_tx = tx[tx['sender'].isin(acc) & tx['case_id'].isin(case_)].drop_duplicates(['sender','case_id'])
        edges[('account','subject_of','case')] = self._make_edge(
            [acc[s] for s in case_tx['sender']], [case_[c] for c in case_tx['case_id']])
        print(f"   Edge account->subject_of->case      : {len(case_tx):,}")

        # ─── 13. bank -> operates_in -> country ────────────────────────────────
        # Each bank -> one country (round-robin assignment for realism)
        cntry_list = sorted(cntry.keys())
        bk_arr     = maps['_bank_arr']
        edges[('bank','operates_in','country')] = self._make_edge(
            list(range(len(bk_arr))),
            [cntry[cntry_list[i % len(cntry_list)]] for i in range(len(bk_arr))])
        print(f"   Edge bank->operates_in->country     : {len(bk_arr):,}")

        # ─── 14. ip_address -> geolocated_in -> country ────────────────────────
        if '_country' in tx.columns:
            ip_cntry = tx[tx['ip_address'].isin(ip) & tx['_country'].isin(cntry)].drop_duplicates('ip_address')
            edges[('ip_address','geolocated_in','country')] = self._make_edge(
                [ip[i] for i in ip_cntry['ip_address']], [cntry[c] for c in ip_cntry['_country']])
            print(f"   Edge ip->geolocated_in->country     : {len(ip_cntry):,}")

        # ─── 15. account -> has_wallet -> crypto_wallet ────────────────────────
        cw_accs  = [cw2a[w] for w in maps['_crypto_wallet_arr'] if cw2a[w] in acc]
        cw_nodes = [cw[w] for w in maps['_crypto_wallet_arr'] if cw2a[w] in acc]
        edges[('account','has_wallet','crypto_wallet')] = self._make_edge(
            [acc[a] for a in cw_accs], cw_nodes)
        print(f"   Edge account->has_wallet->crypto    : {len(cw_accs):,}")

        # ─── 16. atm -> located_in -> bank ─────────────────────────────────────
        atm_bk = tx[tx['receiver'].isin(atm) & tx['bank_id'].isin(bank)].drop_duplicates(['receiver','bank_id'])
        edges[('atm','located_in','bank')] = self._make_edge(
            [atm[r] for r in atm_bk['receiver']], [bank[b] for b in atm_bk['bank_id']])
        print(f"   Edge atm->located_in->bank          : {len(atm_bk):,}")

        # ─── 17. device -> assigned_to -> channel ──────────────────────────────
        dev_ch = tx[tx['device_id'].isin(dev) & tx['channel'].isin(chan)].drop_duplicates(['device_id','channel'])
        edges[('device','assigned_to','channel')] = self._make_edge(
            [dev[d] for d in dev_ch['device_id']], [chan[c] for c in dev_ch['channel']])
        print(f"   Edge device->assigned_to->channel   : {len(dev_ch):,}")

        # ─── 18. merchant -> accepts_crypto -> crypto_wallet ───────────────────
        crypto_merch = tx[tx['merchant_sector']=='Crypto_Exchange'][['receiver','sender']].copy()
        crypto_merch = crypto_merch[crypto_merch['receiver'].isin(merch)]
        cw_arr  = maps['_crypto_wallet_arr']
        acc_set = set(maps['_account_arr'])
        crypto_merch['cw'] = 'CW' + crypto_merch['sender'].str[1:]
        crypto_merch = crypto_merch[crypto_merch['cw'].isin(cw)].drop_duplicates(['receiver','cw'])
        edges[('merchant','accepts_crypto','crypto_wallet')] = self._make_edge(
            [merch[r] for r in crypto_merch['receiver']], [cw[w] for w in crypto_merch['cw']])
        print(f"   Edge merchant->accepts_crypto->cw   : {len(crypto_merch):,}")

        return edges

    # ── Main Build Method ─────────────────────────────────────────────────────

    def build(self, max_partitions: int = 2, save: bool = True) -> HeteroData:
        print("\n" + "=" * 72)
        print("   STEP 8: FULL 14-NODE HETEROGENEOUS GRAPH BUILDER")
        print("=" * 72)

        tx   = self._load_partitions(max_partitions)
        maps = self._build_id_maps(tx)

        print("\n   Building node features...")
        feats = self._build_node_features(tx, maps)

        print("\n   Building labels & masks...")
        labels, tm, vm, em = self._build_labels_masks(tx, maps)

        print("\n   Building 18 edge relations...")
        edges = self._build_edges(tx, maps)

        # ── Assemble HeteroData ───────────────────────────────────────────────
        data = HeteroData()

        # Assign node features
        for nt, feat in feats.items():
            data[nt].x = feat
            data[nt].num_nodes = feat.shape[0]

        # Account labels + masks
        data['account'].y          = labels
        data['account'].train_mask = tm
        data['account'].val_mask   = vm
        data['account'].test_mask  = em

        # Assign edges
        for rel, ei in edges.items():
            data[rel].edge_index = ei

        # ── Save ──────────────────────────────────────────────────────────────
        import os as _os
        # Derive project root from this file's location
        _proj_root = Path(__file__).resolve().parents[2]
        _graph_dir = _proj_root / "data" / "graph"
        _graph_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(_graph_dir / "hetero_graph_14node.pt")
        if save:
            print(f"   [SAVE] Writing to: {out_path}")
            torch.save(data, out_path)
            _sz = _os.path.getsize(out_path)
            print(f"   [SAVE] Confirmed {_sz:,} bytes written to disk.")

        # ── Summary ───────────────────────────────────────────────────────────
        print("\n" + "=" * 72)
        print("   14-NODE GRAPH SUMMARY")
        print("=" * 72)
        print(f"   {'Node Type':<20} {'Count':>10}  {'Feature Dims':>12}")
        print("   " + "-" * 46)
        for nt in sorted(feats.keys()):
            n = feats[nt].shape[0]
            d = feats[nt].shape[1]
            print(f"   {nt:<20} {n:>10,}  {d:>12}")
        print("   " + "-" * 46)
        total_nodes = sum(f.shape[0] for f in feats.values())
        print(f"   {'TOTAL NODES':<20} {total_nodes:>10,}")
        print()
        print(f"   {'Edge Relation':<45} {'Count':>10}")
        print("   " + "-" * 58)
        for rel, ei in edges.items():
            n_edges = ei.shape[1] if ei.dim() == 2 else 0
            lbl = f"({rel[0]}, {rel[1]}, {rel[2]})"
            print(f"   {lbl:<45} {n_edges:>10,}")
        total_edges = sum(e.shape[1] for e in edges.values() if e.dim() == 2)
        print("   " + "-" * 58)
        print(f"   {'TOTAL EDGES':<45} {total_edges:>10,}")
        print("=" * 72)

        if save:
            print(f"\n   [SAVED] {out_path}")

        return data


if __name__ == "__main__":
    builder = FullHeteroGraphBuilder()
    graph   = builder.build(max_partitions=2)
