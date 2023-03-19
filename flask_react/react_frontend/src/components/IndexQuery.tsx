import { useState } from 'react';
import { CircleLoader } from 'react-spinners';
import classNames from 'classnames';
import queryIndex, { ResponseSources } from '../apis/queryIndex';

const IndexQuery = () => {
  const [isLoading, setLoading] = useState(false);
  const [responseText, setResponseText] = useState('');
  const [responseSources, setResponseSources] = useState<ResponseSources[]>([]);

  const handleQuery = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key == 'Enter') {
      setLoading(true);
      queryIndex(e.currentTarget.value).then((response) => {
        setLoading(false);
        setResponseText(response.text);
        setResponseSources(response.sources);
      });
    }
  };

  const sourceElems = responseSources.map((source) => {
    const nodeTitle =
      source.doc_id.length > 28
        ? source.doc_id.substring(0, 28) + '...'
        : source.doc_id;
    const nodeText =
      source.text.length > 150 ? source.text.substring(0, 130) + '...' : source.text;

    return (
      <div key={source.doc_id} className='query__sources__item'>
        <p className='query__sources__item__id'>{nodeTitle}</p>
        <p className='query__sources__item__text'>{nodeText}</p>
        <p className='query__sources__item__footer'>
          Similarity={source.similarity}, start={source.start}, end=
          {source.end}
        </p>
      </div>
    );
  });

  return (
    <div className='query'>
      <div className='query__input'>
        <label htmlFor='query-text'>Ask a question!</label>
        <input
          type='text'
          name='query-text'
          placeholder='Enter a question here'
          onKeyDown={handleQuery}
        ></input>
      </div>

      <CircleLoader
        className={classNames('query__loader', {
          'query__loader--loading': isLoading,
        })}
        color='#00f596'
      />

      <div
        className={classNames('query__results', {
          'query__results--loading': isLoading,
        })}
      >
        <div className='query__sources__item'>
          <p className='query__sources__item__id'>Query Response</p>
        </div>
        {responseText}
      </div>
      <div
        className={classNames('query__sources', {
          'query__sources--loading': isLoading,
        })}
      >
        <div className='query__sources__item'>
          <p className='query__sources__item__id'>Response Sources</p>
        </div>
        {sourceElems}
      </div>
    </div>
  );
};

export default IndexQuery;
