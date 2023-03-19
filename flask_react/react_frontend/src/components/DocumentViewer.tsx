import type { Document } from '../apis/fetchDocuments';

const MAX_TITLE_LENGTH = 32;
const MAX_DOC_LENGTH = 150;

type DocumentViewerProps = {
  documentList: Document[];
};

const DocumentViewer = ({ documentList }: DocumentViewerProps) => {
  const prepend = (array: JSX.Element[], value: JSX.Element): JSX.Element[] => {
    const newArray = array.slice();
    newArray.unshift(value);
    return newArray;
  };

  let documentListElems = documentList.map((document) => {
    // TODO - redo trimming using CSS and text-overflow: ellipsis
    const id =
      document.id.length < MAX_TITLE_LENGTH
        ? document.id
        : document.id.substring(0, MAX_TITLE_LENGTH) + '...';
    const text =
      document.text.length < MAX_DOC_LENGTH
        ? document.text
        : document.text.substring(0, MAX_DOC_LENGTH) + '...';
    return (
      <div key={document.id} className='viewer__list__item'>
        <p className='viewer__list__title'>{id}</p>
        <p className='viewer__list__text'>{text}</p>
      </div>
    );
  });

  // prepend header
  documentListElems = prepend(
    documentListElems,
    <div key='viewer_title' className='viewer__list__item'>
      <p className='viewer__list__header'>My Documents</p>
    </div>,
  );

  console.log(documentListElems);

  return (
    <div className='viewer'>
      <div className='viewer__list'>
        {documentListElems.length > 0 ? (
          documentListElems
        ) : (
          <div>
            <p className='viewer__list__title'>Upload your first document!</p>
            <p className='viewer__list__text'>
              You will see the title and content here
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default DocumentViewer;
