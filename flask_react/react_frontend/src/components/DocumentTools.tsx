import { useEffect, useState } from 'react';
import DocumentUploader from './DocumentUploader';
import DocumentViewer from './DocumentViewer';
import fetchDocuments, { Document } from '../apis/fetchDocuments';

const DocumentTools = () => {
  const [refreshViewer, setRefreshViewer] = useState(false);
  const [documentList, setDocumentList] = useState<Document[]>([]);

  // Get the list on first load
  useEffect(() => {
    fetchDocuments().then((documents) => {
      setDocumentList(documents);
    });
  }, []);

  useEffect(() => {
    if (refreshViewer) {
      setRefreshViewer(false);
      fetchDocuments().then((documents) => {
        setDocumentList(documents);
      });
    }
  }, [refreshViewer]);

  return (
    <div className='document-tools'>
      <DocumentUploader setRefreshViewer={setRefreshViewer} />
      <DocumentViewer documentList={documentList} />
    </div>
  );
};

export default DocumentTools;
