
function PageHeader({ eyebrow, title, description, children }) {
  return (
    <section className="page-heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h2>{title}</h2>
        <p>{description}</p>
      </div>
      {children && <div className="page-heading-actions">{children}</div>}
    </section>
  );
}

export default PageHeader;
