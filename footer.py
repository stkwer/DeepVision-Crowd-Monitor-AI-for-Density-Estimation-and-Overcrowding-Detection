import streamlit as st

def show_footer():
    st.markdown(
        """
        <style>
        .footer-container {
            background-color: #F8F9FA;
            color: #005F5F;
            padding: 15px 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 0.9rem;
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            z-index: 9999;
             gradient: linear-gradient(90deg, #001F3F, #003366)
             text: #E0FFFF
        }
        .footer-container a {
            color: #fbbf24;
            text-decoration: none;
            font-weight: bold;
        }
        .footer-container a:hover {
            text-decoration: underline;
        }
        </style>
        <div class="footer-container">
            © 2025 CrowdVision AI. Developed by <a href="https://github.com/yourusername" target="_blank">Hiren</a>. All rights reserved.
        </div>
        """,
        unsafe_allow_html=True
    )
