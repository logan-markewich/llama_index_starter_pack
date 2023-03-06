import json
import time
from typing import Any, Dict, Optional
import xmltodict
from blobfile import _xml as xml

resp = b"""\
<?xml version="1.0" encoding="utf-8"?>  
<EnumerationResults ContainerName="https://myaccount.blob.core.windows.net/mycontainer">  
  <MaxResults>4</MaxResults>  
  <Blobs>  
    <Blob>  
      <Name>blob1.txt</Name>  
      <Url>https://myaccount.blob.core.windows.net/mycontainer/blob1.txt</Url>  
      <Properties>  
        <Last-Modified>Sun, 27 Sep 2009 18:41:57 GMT</Last-Modified>  
        <Etag>0x8CAE7D55D050B8B</Etag>  
        <Content-Length>8</Content-Length>  
        <Content-Type>text/html</Content-Type>  
        <Content-Encoding />  
        <Content-Language>en-US</Content-Language>  
        <Content-MD5 />  
        <Cache-Control>no-cache</Cache-Control>  
        <BlobType>BlockBlob</BlobType>  
        <LeaseStatus>unlocked</LeaseStatus>  
      </Properties>  
    </Blob>  
    <Blob>  
      <Name>blob2.txt</Name>  
      <Url>https://myaccount.blob.core.windows.net/mycontainer/blob2.txt</Url>  
      <Properties>  
        <Last-Modified>Sun, 27 Sep 2009 12:18:50 GMT</Last-Modified>  
        <Etag>0x8CAE7D55CF6C339</Etag>  
        <Content-Length>100</Content-Length>  
        <Content-Type>text/html</Content-Type>  
        <Content-Encoding />  
        <Content-Language>en-US</Content-Language>  
        <Content-MD5 />  
        <Cache-Control>no-cache</Cache-Control>  
        <BlobType>BlockBlob</BlobType>  
        <LeaseStatus>unlocked</LeaseStatus>  
      </Properties>  
    </Blob>  
    <BlobPrefix>  
      <Name>myfolder/</Name>  
    </BlobPrefix>  
    <Blob>  
      <Name>newblob1.txt</Name>  
      <Url>https://myaccount.blob.core.windows.net/mycontainer/newblob1.txt</Url>  
      <Properties>  
        <Last-Modified>Sun, 27 Sep 2009 16:31:57 GMT</Last-Modified>  
        <Etag>0x8CAE7D55CF6C339</Etag>  
        <Content-Length>25</Content-Length>  
        <Content-Type>text/html</Content-Type>  
        <Content-Encoding />  
        <Content-Language>en-US</Content-Language>  
        <Content-MD5 />  
        <Cache-Control>no-cache</Cache-Control>  
        <BlobType>BlockBlob</BlobType>  
        <LeaseStatus>unlocked</LeaseStatus>  
      </Properties>  
    </Blob>  
    <BlobPrefix>  
      <Name>myfolder2/</Name>  
    </BlobPrefix>  
  </Blobs>  
  <NextMarker>newblob2.txt</NextMarker>  
</EnumerationResults>  
"""


def remove_attributes(d: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for k, v in d.items():
        if k.startswith("@"):
            continue
        if isinstance(v, dict):
            v = remove_attributes(v)
        result[k] = v
    return result


def xmltodict_parse(data: bytes) -> Dict[str, Any]:
    parsed = xmltodict.parse(data.decode("utf8"))
    # we don't support attributes in our parser since we don't use them anyway
    return remove_attributes(parsed)


def xmltodict_unparse(d: Dict[str, Any]) -> bytes:
    return xmltodict.unparse(d).encode("utf8")


def test_parse():
    ref = xmltodict_parse(resp)
    actual = xml.parse(resp, repeated_tags={"Blob", "BlobPrefix"})
    json_ref = json.dumps(ref, sort_keys=True, indent=" ")
    json_actual = json.dumps(actual, sort_keys=True, indent=" ")
    print(json_ref)
    print(json_actual)
    assert json_ref == json_actual


def test_unparse():
    body = {"BlockList": {"Latest": [str(i) for i in range(100)]}}
    ref = xmltodict_unparse(body)
    actual = xml.unparse(body)
    print(ref)
    print(actual)
    assert ref == actual


def test_roundtrip():
    ref_parsed = xmltodict_parse(resp)
    ref_unparsed = xmltodict_unparse(ref_parsed)

    actual_parsed = xml.parse(ref_unparsed, repeated_tags={"Blob", "BlobPrefix"})
    assert ref_parsed == actual_parsed

    actual_unparsed = xml.unparse(ref_parsed)
    print()
    print(ref_unparsed)
    print(actual_unparsed)
    assert ref_unparsed == actual_unparsed

    assert ref_unparsed == xml.unparse(
        xml.parse(xml.unparse(ref_parsed), repeated_tags={"Blob", "BlobPrefix"})
    )


def main():
    # benchmarking
    doc = xmltodict_parse(resp)
    doc2 = doc.copy()
    doc2["EnumerationResults"]["Blobs"]["Blob"] = (
        doc2["EnumerationResults"]["Blobs"]["Blob"] * 300
    )
    expanded_resp_utf8 = xmltodict_unparse(doc2)

    start = time.perf_counter()
    for _ in range(100):
        xmltodict.parse(expanded_resp)
    end = time.perf_counter()
    print(f"xmltodict parse elapsed {end - start}")

    start = time.perf_counter()
    for _ in range(100):
        lxml_parse(expanded_resp_utf8, repeated_tags={"Blob", "BlobPrefix"})
    end = time.perf_counter()
    print(f"lxml parse elapsed {end - start}")

    start = time.perf_counter()
    for _ in range(100):
        xmltodict.unparse(doc2)
    end = time.perf_counter()
    print(f"xmltodict unparse elapsed {end - start}")

    start = time.perf_counter()
    for _ in range(100):
        lxml_unparse(doc2)
    end = time.perf_counter()
    print(f"lxml unparse elapsed {end - start}")


if __name__ == "__main__":
    main()
