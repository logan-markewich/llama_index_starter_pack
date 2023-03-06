from typing import Union, Dict, Iterable, Optional, ByteString

from Cryptodome.Cipher._mode_ecb import EcbMode
from Cryptodome.Cipher._mode_cbc import CbcMode
from Cryptodome.Cipher._mode_cfb import CfbMode
from Cryptodome.Cipher._mode_ofb import OfbMode
from Cryptodome.Cipher._mode_ctr import CtrMode
from Cryptodome.Cipher._mode_openpgp import OpenPgpMode
from Cryptodome.Cipher._mode_eax import EaxMode

CASTMode = int

MODE_ECB: CASTMode
MODE_CBC: CASTMode
MODE_CFB: CASTMode
MODE_OFB: CASTMode
MODE_CTR: CASTMode
MODE_OPENPGP: CASTMode
MODE_EAX: CASTMode

def new(key: ByteString,
        mode: CASTMode,
        iv : Optional[ByteString] = ...,
        IV : Optional[ByteString] = ...,
        nonce : Optional[ByteString] = ...,
        segment_size : int = ...,
        mac_len : int = ...,
        initial_value : Union[int, ByteString] = ...,
        counter : Dict = ...) -> \
        Union[EcbMode, CbcMode, CfbMode, OfbMode, CtrMode, OpenPgpMode]: ...

block_size: int
key_size : Iterable[int]
