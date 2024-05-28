import numpy as np
import streamlit as st
from PIL import Image
import test


def upload_image_sidebar():
    st.sidebar.title('上传图片')
    uploaded_file = st.sidebar.file_uploader("选择你需要上传的图片", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        return image


def recognize_image(image, confidence):
    results = test.main(uploaded_image=image, confidence=confidence)
    return results


def recognize_sidebar(image, confidence):
    st.sidebar.title('动作')
    if st.sidebar.button('识别图片'):
        return recognize_image(image, confidence)


def confidence_slider_sidebar():
    st.sidebar.title('相关配置')
    confidence = st.sidebar.slider('选择置信度:', 0.0, 1.0, 0.6, 0.01)
    return confidence


def display_images(original_image, processed_image):
    st.title('麦穗识别')
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption='Original Image', use_column_width=True)
    with col2:
        st.image(processed_image, caption='Processed Image', use_column_width=True)


def display_results_main(results):
    if results['boxes_num'] > 0:
        st.success(f"麦穗数量为: {results['boxes_num']}")
    else:
        st.error('暂未识别到目标，请调整置信度或切换图片后再次尝试')


def main():
    st.set_page_config(page_title='Wheat Visualization App')

    image = upload_image_sidebar()
    if image is None:
        st.info('请上传图片')
        st.stop()
    confidence = confidence_slider_sidebar()
    results = recognize_sidebar(image, confidence)

    if results is not None:
        display_images(image, results['original_image'])
        display_results_main(results)
    else:
        display_images(image, image)
        st.info('请点击左侧按钮进行识别')


if __name__ == '__main__':
    main()
