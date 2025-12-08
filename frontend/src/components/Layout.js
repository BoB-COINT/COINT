// Layout.js
import { useLocation } from 'react-router-dom';
import SearchBar from './SearchBar';
import Sidebar from './Sidebar';
import './Sidebar.css';

function Layout({ children }) {
  const location = useLocation();
  const isHome = location.pathname === '/';

  return (
    <div
      className="layout-root"
      style={{ display: 'flex', minHeight: '100vh', backgroundColor: '#171821' }}
    >
      <Sidebar />
      <div className="layout-main" style={{ flex: 1, backgroundColor: '#171821' }}>
        {/* Home(/)에서는 상단 공통 SearchBar 숨기기 */}
        {!isHome && <SearchBar />}

        <main
          style={{
            padding: '20px',
            backgroundColor: '#171821',
          }}
        >
          {children}
        </main>
      </div>
    </div>
  );
}

export default Layout;
