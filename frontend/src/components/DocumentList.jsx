/**
 * Sidebar list of uploaded documents.
 *
 * Props:
 *   documents  {string[]}        - filenames returned by GET /documents
 *   selected   {string|null}     - currently selected filename; null = all docs
 *   onSelect   {(name|null)=>void}
 */
export default function DocumentList({ documents, selected, onSelect }) {
  if (!documents.length) {
    return <p className="doc-list-empty">No documents yet.<br />Upload a PDF above.</p>
  }

  return (
    <div className="doc-list">
      {selected && (
        <button className="doc-list-all" onClick={() => onSelect(null)}>
          Show all documents
        </button>
      )}
      <ul className="doc-list-items">
        {documents.map(name => (
          <li
            key={name}
            className={`doc-list-item${selected === name ? ' selected' : ''}`}
            onClick={() => onSelect(selected === name ? null : name)}
            title={name}
          >
            <span className="doc-list-name">{name}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
