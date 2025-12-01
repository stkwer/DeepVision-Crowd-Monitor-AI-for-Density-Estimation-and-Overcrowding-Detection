import streamlit as st

def show_header():
    # Custom CSS for header
    st.markdown(
        """
        <style>
        .header-container {
            background-color: #1e3a8a;
            padding: 10px 20px;
            border-radius: 10px;
            color: white;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header-title {
            font-size: 1.5rem;
            font-weight: 600;
        }
        .nav-button button {
            background-color: #2563eb;
            color: white;
            border: none;
            padding: 6px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1rem;
            margin-left: 5px;
        }
        .nav-button button:hover {
            background-color: #1d4ed8;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Layout: Title left, buttons right
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown('<div class="header-title"> </div>', unsafe_allow_html=True)
    
    with col2:
        nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)
        with nav_col1:
            if st.button("Home", key="btn_home_header"):
                st.session_state["page"] = "home"
        with nav_col2:
            if st.button("Login", key="btn_login_header"):
                st.session_state["page"] = "login"
        with nav_col3:
            if st.button("Signup", key="btn_signup_header"):
                st.session_state["page"] = "signup"
        with nav_col4:
            if st.button("Dashboard", key="btn_dashboard_header"):
                st.session_state["page"] = "dashboard"
        with nav_col5:
            if st.button("About", key="btn_about_header"):
                st.session_state["page"] = "about"
