/**
 * Admin Dashboard JavaScript
 * Handles all dashboard functionality, data loading, and user interactions
 */

class AdminDashboard {
    constructor() {
        this.charts = {};
        this.currentData = {};
        this.selectedSources = new Set();
        this.init();
    }

    async init() {
        console.log('Initializing Admin Dashboard...');
        this.setupEventListeners();
        await this.loadInitialData();
        this.setupTooltips();
    }

    setupEventListeners() {
        // Tab switching
        document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
            tab.addEventListener('shown.bs.tab', (e) => {
                this.onTabChange(e.target.getAttribute('data-bs-target'));
            });
        });

        // Delete confirmation
        document.getElementById('confirmDeleteBtn').addEventListener('click', () => {
            this.confirmDelete();
        });
        
        // Create user form - set up later when form is loaded
        this.setupCreateUserForm();
    }

    setupCreateUserForm() {
        // This will be called when the user management tab is loaded
        setTimeout(() => {
            const createUserForm = document.getElementById('createUserForm');
            if (createUserForm) {
                createUserForm.addEventListener('submit', (e) => {
                    e.preventDefault();
                    this.handleCreateUser();
                });
            }
        }, 1000);
    }

    setupTooltips() {
        // Initialize Bootstrap tooltips
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
    }

    showNotification(message, type = 'info') {
        const sharedNotify = window?.DI?.utilities?.notifications?.showNotification
            || (typeof window.showNotification === 'function' ? window.showNotification : null);

        if (typeof sharedNotify === 'function') {
            sharedNotify(message, type);
            return;
        }

        if (typeof alert === 'function') {
            alert(`${type.toUpperCase()}: ${message}`);
        } else {
            console.log(`[${type.toUpperCase()}] ${message}`);
        }
    }

    async loadUsers(page = 1, limit = 10) {
        try {
            console.log('Loading users...');
            
            // Show loading state
            const tbody = document.getElementById('usersTableBody');
            if (tbody) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="8" class="text-center py-4">
                            <div class="spinner-border spinner-border-sm me-2" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                            Loading users...
                        </td>
                    </tr>
                `;
            }
            
            // Get filter values
            const search = document.getElementById('userSearchInput')?.value || '';
            const statusFilter = document.getElementById('userStatusFilter')?.value;
            const roleFilter = document.getElementById('userRoleFilter')?.value;
            
            // Build query parameters
            const params = new URLSearchParams({
                skip: (page - 1) * limit,
                limit: limit
            });
            
            if (search) params.append('search', search);
            if (statusFilter !== '') params.append('is_active', statusFilter);
            if (roleFilter !== '') params.append('is_admin', roleFilter);
            
            // Use correct endpoint
            const response = await fetch(`/admin/api/users?${params}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            console.log('Users API response:', result);
            
            if (result.success && result.users) {
                this.renderUsersTable(result.users);
                this.updateUsersPagination(result, page, limit);
            } else {
                console.error('Failed to load users:', result.error || 'Unknown error');
                this.showNotification('Failed to load users: ' + (result.error || 'Unknown error'), 'error');
                if (tbody) {
                    tbody.innerHTML = `
                        <tr>
                            <td colspan="8" class="text-center text-danger py-4">
                                <i class="fas fa-exclamation-triangle me-2"></i>
                                Failed to load users
                            </td>
                        </tr>
                    `;
                }
            }
        } catch (error) {
            console.error('Error loading users:', error);
            this.showNotification('Error loading users: ' + error.message, 'error');
            const tbody = document.getElementById('usersTableBody');
            if (tbody) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="8" class="text-center text-danger py-4">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            Error loading users: ${error.message}
                        </td>
                    </tr>
                `;
            }
        }
    }

    renderUsersTable(users) {
        const tbody = document.getElementById('usersTableBody');
        if (!tbody) {
            console.error('Users table body not found');
            return;
        }
        
        if (users.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="8" class="text-center py-4">
                        <i class="fas fa-users fa-2x text-muted"></i>
                        <p class="mt-2 text-muted">No users found</p>
                    </td>
                </tr>
            `;
            return;
        }

        const rows = users.map(user => `
            <tr>
                <td>
                    <div class="d-flex align-items-center">
                        <div class="avatar-sm me-3">
                            <div class="avatar-title bg-primary text-white rounded-circle">
                                ${(user.username || 'U').charAt(0).toUpperCase()}
                            </div>
                        </div>
                        <div>
                            <h6 class="mb-0">${this.escapeHtml(user.username || 'Unknown')}</h6>
                            ${user.full_name ? `<small class="text-muted">${this.escapeHtml(user.full_name)}</small>` : ''}
                        </div>
                    </div>
                </td>
                <td>
                    <span class="text-dark">${this.escapeHtml(user.email || 'No email')}</span>
                </td>
                <td>
                    <span class="badge ${user.is_verified ? 'bg-success' : 'bg-warning'}">
                        ${user.is_verified ? 'Verified' : 'Pending'}
                    </span>
                </td>
                <td>
                    <span class="badge ${user.is_active ? 'bg-success' : 'bg-secondary'}">
                        ${user.is_active ? 'Active' : 'Inactive'}
                    </span>
                </td>
                <td>
                    <span class="badge ${user.is_admin ? 'bg-danger' : 'bg-primary'}">
                        ${user.is_admin ? 'Admin' : 'User'}
                    </span>
                </td>
                <td>
                    <small class="text-muted">${this.formatDate(user.last_login) || 'Never'}</small>
                </td>
                <td>
                    <small class="text-muted">${this.formatDate(user.created_at) || 'N/A'}</small>
                </td>
                <td>
                    <div class="d-flex gap-1">
                        <button class="btn btn-sm btn-outline-primary" 
                                onclick="window.adminDashboard.editUser('${user.id}')" 
                                title="Edit User">
                            <i class="fas fa-edit"></i>
                        </button>
                        <button class="btn btn-sm btn-outline-${user.is_active ? 'warning' : 'success'}" 
                                onclick="window.adminDashboard.toggleUserStatus('${user.id}', '${this.escapeHtml(user.username)}', ${user.is_active})" 
                                title="${user.is_active ? 'Deactivate' : 'Activate'} User">
                            <i class="fas fa-${user.is_active ? 'pause' : 'play'}"></i>
                        </button>
                        <button class="btn btn-sm btn-outline-danger" 
                                onclick="window.adminDashboard.deleteUser('${user.id}', '${this.escapeHtml(user.username)}')" 
                                title="Delete User">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </td>
            </tr>
        `).join('');
        
        tbody.innerHTML = rows;
    }

    // Alias for backward compatibility
    enhancedRenderUsersTable(users) {
        return this.renderUsersTable(users);
    }

    updateUsersPagination(result, currentPage, limit) {
        const paginationInfo = document.getElementById('userPaginationInfo');
        if (paginationInfo && result.users && result.users.length > 0) {
            const start = (currentPage - 1) * limit + 1;
            const end = Math.min(start + result.users.length - 1, result.total || result.users.length);
            paginationInfo.textContent = `${start}-${end} of ${result.total || result.users.length}`;
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    async toggleUserStatus(userId, username, currentStatus) {
        const action = currentStatus ? 'deactivate' : 'activate';
        const actionPast = currentStatus ? 'deactivated' : 'activated';
        
        if (!confirm(`Are you sure you want to ${action} user "${username}"?`)) {
            return;
        }

        try {
            const response = await fetch(`/admin/api/users/${userId}`, {
                method: 'PATCH',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    is_active: !currentStatus
                })
            });

            const result = await response.json();

            if (result.success) {
                this.showNotification(`User "${username}" has been ${actionPast}`, 'success');
                // Reload the users table to reflect the change
                await this.loadUsers();
            } else {
                this.showNotification(`Failed to ${action} user: ${result.error}`, 'error');
            }
        } catch (error) {
            console.error(`Error ${action}ing user:`, error);
            this.showNotification(`Error ${action}ing user`, 'error');
        }
    }

    async editUser(userId) {
        try {
            // First, get the user data
            const response = await fetch(`/admin/api/users?skip=0&limit=1000`);
            const result = await response.json();
            
            if (!result.success) {
                this.showNotification('Failed to load user data', 'error');
                return;
            }
            
            const user = result.users.find(u => u.id == userId);
            if (!user) {
                this.showNotification('User not found', 'error');
                return;
            }
            
            // Create and show the edit user modal
            this.showEditUserModal(user);
            
        } catch (error) {
            console.error('Error loading user for editing:', error);
            this.showNotification('Error loading user data', 'error');
        }
    }

    showEditUserModal(user) {
        // Create modal HTML
        const modalHTML = `
            <div class="modal fade" id="editUserModal" tabindex="-1" aria-labelledby="editUserModalLabel" aria-hidden="true">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title" id="editUserModalLabel">
                                <i class="fas fa-user-edit me-2"></i>Edit User: ${this.escapeHtml(user.username)}
                            </h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <form id="editUserForm">
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="mb-3">
                                            <label for="editUsername" class="form-label">Username</label>
                                            <input type="text" class="form-control" id="editUsername" value="${this.escapeHtml(user.username)}" required>
                                            <div class="invalid-feedback"></div>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="mb-3">
                                            <label for="editEmail" class="form-label">Email</label>
                                            <input type="email" class="form-control" id="editEmail" value="${this.escapeHtml(user.email)}" required>
                                            <div class="invalid-feedback"></div>
                                        </div>
                                    </div>
                                </div>
                                <div class="row">
                                    <div class="col-md-12">
                                        <div class="mb-3">
                                            <label for="editFullName" class="form-label">Full Name</label>
                                            <input type="text" class="form-control" id="editFullName" value="${this.escapeHtml(user.full_name || '')}">
                                        </div>
                                    </div>
                                </div>
                                <div class="row">
                                    <div class="col-md-4">
                                        <div class="form-check form-switch mb-3">
                                            <input class="form-check-input" type="checkbox" id="editIsAdmin" ${user.is_admin ? 'checked' : ''}>
                                            <label class="form-check-label" for="editIsAdmin">
                                                Administrator privileges
                                            </label>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="form-check form-switch mb-3">
                                            <input class="form-check-input" type="checkbox" id="editIsActive" ${user.is_active ? 'checked' : ''}>
                                            <label class="form-check-label" for="editIsActive">
                                                Account active
                                            </label>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="form-check form-switch mb-3">
                                            <input class="form-check-input" type="checkbox" id="editIsVerified" ${user.is_verified ? 'checked' : ''}>
                                            <label class="form-check-label" for="editIsVerified">
                                                Email verified
                                            </label>
                                        </div>
                                    </div>
                                </div>
                            </form>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                            <button type="button" class="btn btn-primary" onclick="window.adminDashboard.saveUserChanges(${user.id})">
                                <i class="fas fa-save me-2"></i>Save Changes
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Remove any existing modal
        const existingModal = document.getElementById('editUserModal');
        if (existingModal) {
            existingModal.remove();
        }
        
        // Add modal to DOM
        document.body.insertAdjacentHTML('beforeend', modalHTML);
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('editUserModal'));
        modal.show();
        
        // Clean up when modal is hidden
        document.getElementById('editUserModal').addEventListener('hidden.bs.modal', function () {
            this.remove();
        });
    }

    async saveUserChanges(userId) {
        try {
            const form = document.getElementById('editUserForm');
            const formData = new FormData(form);
            
            // Get form values
            const updateData = {
                username: document.getElementById('editUsername').value.trim(),
                email: document.getElementById('editEmail').value.trim(),
                full_name: document.getElementById('editFullName').value.trim() || null,
                is_admin: document.getElementById('editIsAdmin').checked,
                is_active: document.getElementById('editIsActive').checked,
                is_verified: document.getElementById('editIsVerified').checked
            };
            
            // Validate required fields
            if (!updateData.username || !updateData.email) {
                this.showNotification('Username and email are required', 'error');
                return;
            }
            
            const response = await fetch(`/admin/api/users/${userId}`, {
                method: 'PATCH',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(updateData)
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showNotification('User updated successfully', 'success');
                
                // Close modal
                const modal = bootstrap.Modal.getInstance(document.getElementById('editUserModal'));
                modal.hide();
                
                // Reload users table
                await this.loadUsers();
            } else {
                this.showNotification(`Failed to update user: ${result.detail || 'Unknown error'}`, 'error');
            }
        } catch (error) {
            console.error('Error saving user changes:', error);
            this.showNotification('Error saving user changes', 'error');
        }
    }

    async deleteUser(userId, username) {
        if (!confirm(`Are you sure you want to permanently delete user "${username}"? This action cannot be undone.`)) {
            return;
        }

        try {
            const response = await fetch(`/admin/api/users/${userId}`, {
                method: 'DELETE'
            });

            const result = await response.json();

            if (result.success) {
                this.showNotification(`User "${username}" has been deleted`, 'success');
                await this.loadUsers();
            } else {
                this.showNotification(`Failed to delete user: ${result.error}`, 'error');
            }
        } catch (error) {
            console.error('Error deleting user:', error);
            this.showNotification('Error deleting user', 'error');
        }
    }

    async handleCreateUser() {
        try {
            // Get form values
            const username = document.getElementById('newUsername').value.trim();
            const email = document.getElementById('newEmail').value.trim();
            const password = document.getElementById('newPassword').value;
            const fullName = document.getElementById('newFullName').value.trim();
            const isAdmin = document.getElementById('newIsAdmin').checked;
            const isVerified = document.getElementById('newIsVerified').checked;
            
            // Validate required fields
            if (!username || !email || !password) {
                this.showNotification('Username, email, and password are required', 'error');
                return;
            }
            
            if (password.length < 6) {
                this.showNotification('Password must be at least 6 characters long', 'error');
                return;
            }
            
            // Create user data object
            const userData = {
                username,
                email,
                password,
                full_name: fullName || null,
                is_admin: isAdmin,
                is_verified: isVerified
            };
            
            const response = await fetch('/admin/api/users', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(userData)
            });
            
            const result = await response.json();
            
            if (response.ok && result.success) {
                this.showNotification(`User "${username}" created successfully`, 'success');
                
                // Reset form
                this.resetCreateUserForm();
                
                // Reload users table
                await this.loadUsers();
                
                // Switch to user list tab
                const userListTab = document.getElementById('user-list-tab');
                if (userListTab) {
                    userListTab.click();
                }
            } else {
                // Handle both API errors and HTTP errors
                const errorMessage = result.detail || result.message || 'Unknown error';
                this.showNotification(`Failed to create user: ${errorMessage}`, 'error');
            }
            
        } catch (error) {
            console.error('Error creating user:', error);
            this.showNotification('Error creating user', 'error');
        }
    }

    resetCreateUserForm() {
        const form = document.getElementById('createUserForm');
        if (form) {
            form.reset();
            
            // Clear any validation states
            form.querySelectorAll('.is-invalid').forEach(el => {
                el.classList.remove('is-invalid');
            });
            form.querySelectorAll('.is-valid').forEach(el => {
                el.classList.remove('is-valid');
            });
        }
    }

    async loadInitialData() {
        await this.loadDataIngestionStats();
        await this.loadDataSources();
        await this.loadUserStats();
        await this.loadMaintenanceStatus();
    }

    async onTabChange(tabTarget) {
        console.log('Tab changed to:', tabTarget);
        
        if (tabTarget === '#data-ingestion') {
            await this.loadDataIngestionStats();
            await this.loadDataSources();
        } else if (tabTarget === '#user-management') {
            await this.loadUserStats();
            await this.loadUsers();
            
            // Load stats data and update charts
            const statsData = await this.getUserStatsData();
            if (statsData) {
                this.updateStatisticsAnalyticsTab(statsData);
                this.createUserCharts(statsData);
            }
            
            // Set up create user form listener after a short delay to ensure DOM is loaded
            setTimeout(() => {
                this.setupUserManagementListeners();
            }, 500);
        } else if (tabTarget === '#system-info') {
            await this.loadSystemInfo();
        } else if (tabTarget === '#app-logs') {
            await this.loadAppLogs();
        } else if (tabTarget === '#system-maintenance') {
            await this.loadMaintenanceStatus();
        }
    }

    setupUserManagementListeners() {
        const createUserForm = document.getElementById('createUserForm');
        if (createUserForm && !createUserForm.hasAttribute('data-listener-added')) {
            createUserForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleCreateUser();
            });
            createUserForm.setAttribute('data-listener-added', 'true');
        }

        // Set up search and filter listeners
        const searchInput = document.getElementById('userSearchInput');
        const statusFilter = document.getElementById('userStatusFilter');
        const roleFilter = document.getElementById('userRoleFilter');

        if (searchInput && !searchInput.hasAttribute('data-listener-added')) {
            searchInput.addEventListener('input', () => {
                clearTimeout(this.searchTimeout);
                this.searchTimeout = setTimeout(() => {
                    this.loadUsers();
                }, 500);
            });
            searchInput.setAttribute('data-listener-added', 'true');
        }

        if (statusFilter && !statusFilter.hasAttribute('data-listener-added')) {
            statusFilter.addEventListener('change', () => {
                this.loadUsers();
            });
            statusFilter.setAttribute('data-listener-added', 'true');
        }

        if (roleFilter && !roleFilter.hasAttribute('data-listener-added')) {
            roleFilter.addEventListener('change', () => {
                this.loadUsers();
            });
            roleFilter.setAttribute('data-listener-added', 'true');
        }

        // Set up listeners for user management sub-tabs
        this.setupUserManagementSubTabs();
    }

    setupUserManagementSubTabs() {
        const userStatsTab = document.getElementById('user-stats-tab');
        if (userStatsTab && !userStatsTab.hasAttribute('data-listener-added')) {
            userStatsTab.addEventListener('shown.bs.tab', async () => {
                console.log('Statistics & Analytics tab activated');
                await this.loadUserStatsForAnalytics();
            });
            // Also try the pill event in case it's using pills
            userStatsTab.addEventListener('click', async () => {
                console.log('Statistics & Analytics tab clicked');
                // Small delay to ensure tab is fully shown
                setTimeout(async () => {
                    await this.loadUserStatsForAnalytics();
                }, 100);
            });
            userStatsTab.setAttribute('data-listener-added', 'true');
        }
    }

    async loadUserStatsForAnalytics() {
        try {
            const stats = await this.getUserStatsData();
            if (stats) {
                this.updateStatisticsAnalyticsTab(stats);
                this.createUserCharts(stats);
            }
        } catch (error) {
            console.error('Error loading user stats for analytics:', error);
            this.showNotification('Failed to load analytics data', 'error');
        }
    }

    // =============================================================================
    // Data Ingestion Management
    // =============================================================================

    async loadDataIngestionStats() {
        try {
            console.log('Loading data ingestion stats...');
            const response = await fetch('/admin/api/data-ingestion/stats');
            const result = await response.json();

            if (result.success) {
                this.updateDataIngestionStats(result.stats);
                this.updateChartsData(result.stats);
            } else {
                this.showNotification('Error loading statistics: ' + result.error, 'error');
            }
        } catch (error) {
            console.error('Error loading data ingestion stats:', error);
            this.showNotification('Failed to load statistics', 'error');
        }
    }

    updateDataIngestionStats(stats) {
        // Update stat cards
        document.getElementById('totalSources').textContent = stats.total_sources.toLocaleString();
        document.getElementById('activeSources').textContent = stats.active_sources.toLocaleString();
        document.getElementById('totalSize').textContent = stats.total_size_formatted || '0 B';
        document.getElementById('totalRows').textContent = stats.total_rows.toLocaleString();
    }

    updateChartsData(stats) {
        // Sources by Type Chart
        if (stats.sources_by_type && Object.keys(stats.sources_by_type).length > 0) {
            this.createPieChart('sourcesByTypeChart', 'Sources by Type', stats.sources_by_type);
        } else {
            document.getElementById('sourcesByTypeChart').innerHTML = 
                '<div class="text-center text-muted py-4"><i class="fas fa-chart-pie fa-2x"></i><p class="mt-2">No data available</p></div>';
        }

        // Sources by Category Chart
        if (stats.sources_by_category && Object.keys(stats.sources_by_category).length > 0) {
            this.createPieChart('sourcesByCategoryChart', 'Sources by Category', stats.sources_by_category);
        } else {
            document.getElementById('sourcesByCategoryChart').innerHTML = 
                '<div class="text-center text-muted py-4"><i class="fas fa-chart-donut fa-2x"></i><p class="mt-2">No data available</p></div>';
        }
    }

    createPieChart(containerId, title, data) {
        const container = document.getElementById(containerId);
        
        // Clear existing chart
        if (this.charts[containerId]) {
            this.charts[containerId].destroy();
        }

        // Create canvas
        container.innerHTML = `<canvas id="${containerId}Canvas"></canvas>`;
        const ctx = document.getElementById(`${containerId}Canvas`).getContext('2d');

        const labels = Object.keys(data);
        const values = Object.values(data);
        const colors = this.generateColors(labels.length);

        this.charts[containerId] = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: colors,
                    borderWidth: 2,
                    borderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((context.parsed * 100) / total).toFixed(1);
                                return `${context.label}: ${context.parsed} (${percentage}%)`;
                            }
                        }
                    }
                },
                cutout: '60%'
            }
        });
    }

    generateColors(count) {
        const colors = [
            '#2563eb', '#059669', '#d97706', '#dc2626', '#0891b2',
            '#7c3aed', '#be185d', '#047857', '#b45309', '#9333ea'
        ];
        
        const result = [];
        for (let i = 0; i < count; i++) {
            result.push(colors[i % colors.length]);
        }
        return result;
    }

    async loadDataSources() {
        try {
            console.log('Loading data sources...');
            const response = await fetch('/admin/api/data-ingestion/sources');
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            console.log('Data sources API response:', result);

            if (result.success) {
                this.currentData.dataSources = result.sources;
                console.log('About to render table with sources:', result.sources);
                this.renderDataSourcesTable(result.sources);
            } else {
                console.error('API returned error:', result.error);
                this.showNotification('Error loading data sources: ' + result.error, 'error');
            }
        } catch (error) {
            console.error('Error loading data sources:', error);
            this.showNotification('Failed to load data sources', 'error');
            
            // Show error in table
            const tbody = document.getElementById('dataSourcesTableBody');
            tbody.innerHTML = `
                <tr>
                    <td colspan="10" class="text-center py-4 text-danger">
                        <i class="fas fa-exclamation-triangle fa-2x"></i>
                        <p class="mt-2">Error loading data sources: ${error.message}</p>
                    </td>
                </tr>
            `;
        }
    }

    renderDataSourcesTable(sources) {
        const tbody = document.getElementById('dataSourcesTableBody');
        
        if (!tbody) {
            console.error('Table body element not found!');
            return;
        }
        
        // Clear any existing content
        tbody.innerHTML = '';
        
        console.log('Rendering data sources table with:', sources.length, 'sources');
        
        if (sources.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="10" class="text-center py-4">
                        <i class="fas fa-database fa-2x text-muted"></i>
                        <p class="mt-2 text-muted">No data sources found</p>
                    </td>
                </tr>
            `;
            return;
        }

        const tableRows = sources.map(source => {
            const safeSource = {
                id: source.id || 'N/A',
                source_id: source.source_id || 'N/A',
                name: source.name || 'Unnamed',
                type: source.type || 'unknown',
                category: source.category || 'N/A',
                created_by: source.created_by || 'Unknown',
                created_at: source.created_at || null,
                file_size_formatted: source.file_size_formatted || 'N/A',
                estimated_rows: source.estimated_rows,
                quality_score: source.quality_score
            };
            
            return `
                <tr data-source-id="${safeSource.id}">
                    <td>
                        <input type="checkbox" class="source-checkbox" value="${safeSource.id}" 
                               onchange="window.adminDashboard.toggleSourceSelection('${safeSource.id}', this.checked)">
                    </td>
                    <td>
                        <div class="d-flex align-items-center">
                            <div class="source-icon me-2">
                                <i class="fas ${this.getSourceIcon(safeSource.type)} text-primary"></i>
                            </div>
                            <div>
                                <div class="fw-medium">${this.escapeHtml(safeSource.name)}</div>
                                <small class="text-muted">Source ID: ${safeSource.source_id}</small>
                            </div>
                        </div>
                    </td>
                    <td><span class="text-muted font-monospace">${safeSource.id}</span></td>
                    <td><span class="badge bg-light text-dark">${safeSource.type}</span></td>
                    <td><span class="badge bg-secondary">${safeSource.category}</span></td>
                    <td><span class="text-muted">${safeSource.created_by}</span></td>
                    <td><span class="text-muted">${this.formatDate(safeSource.created_at)}</span></td>
                    <td><span class="text-muted">${safeSource.file_size_formatted}</span></td>
                    <td><span class="text-muted">${(safeSource.estimated_rows !== null && safeSource.estimated_rows !== undefined) ? safeSource.estimated_rows.toLocaleString() : 'N/A'}</span></td>
                    <td>${(safeSource.quality_score !== null && safeSource.quality_score !== undefined) ? 
                            `<span class="badge ${this.getQualityBadgeClass(safeSource.quality_score)}">${Math.round(safeSource.quality_score * 100)}%</span>` 
                            : '<span class="text-muted">N/A</span>'
                        }</td>
                    <td>
                        <div class="btn-group btn-group-sm">
                            <button class="btn btn-outline-primary btn-sm" onclick="window.adminDashboard.previewSource('${safeSource.source_id}')" title="Preview">
                                <i class="fas fa-eye"></i>
                            </button>
                            <button class="btn btn-outline-danger btn-sm" onclick="window.adminDashboard.deleteSource('${safeSource.id}')" title="Delete">
                                <i class="fas fa-trash-alt"></i>
                            </button>
                        </div>
                    </td>
                </tr>
            `;
        });
        
        tbody.innerHTML = tableRows.join('');
        console.log('Table rendering completed with', tableRows.length, 'rows');
    }

    getSourceIcon(sourceType) {
        const icons = {
            'csv': 'fa-file-csv',
            'excel': 'fa-file-excel',
            'json': 'fa-file-code',
            'txt': 'fa-file-alt',
            'file': 'fa-file',
            'postgres': 'fa-database',
            'postgresql': 'fa-database',
            'mysql': 'fa-database',
            'sqlite': 'fa-database',
            'database': 'fa-database',
            'api': 'fa-cloud',
            'rest': 'fa-cloud',
            'graphql': 'fa-cloud',
            'http': 'fa-cloud'
        };
        return icons[sourceType] || 'fa-file';
    }

    getQualityBadgeClass(score) {
        if (score >= 0.8) return 'bg-success';
        if (score >= 0.6) return 'bg-warning';
        return 'bg-danger';
    }

    formatDate(dateStr) {
        if (!dateStr) return 'N/A';
        try {
            const date = new Date(dateStr);
            return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        } catch {
            return 'Invalid Date';
        }
    }

    toggleSourceSelection(sourceId, selected) {
        if (selected) {
            this.selectedSources.add(sourceId);
        } else {
            this.selectedSources.delete(sourceId);
        }
        
        // Update select all checkbox
        const selectAllCheckbox = document.getElementById('selectAllSources');
        const totalCheckboxes = document.querySelectorAll('.source-checkbox').length;
        
        if (this.selectedSources.size === 0) {
            selectAllCheckbox.indeterminate = false;
            selectAllCheckbox.checked = false;
        } else if (this.selectedSources.size === totalCheckboxes) {
            selectAllCheckbox.indeterminate = false;
            selectAllCheckbox.checked = true;
        } else {
            selectAllCheckbox.indeterminate = true;
            selectAllCheckbox.checked = false;
        }
    }

    toggleAllSources() {
        const selectAllCheckbox = document.getElementById('selectAllSources');
        const checkboxes = document.querySelectorAll('.source-checkbox');
        
        checkboxes.forEach(checkbox => {
            checkbox.checked = selectAllCheckbox.checked;
            this.toggleSourceSelection(checkbox.value, checkbox.checked);
        });
    }

    async deleteSource(sourceId) {
        console.log('DELETE SOURCE CALLED with ID:', sourceId);
        
        if (!this.currentData.dataSources || this.currentData.dataSources.length === 0) {
            console.error('No data sources available');
            this.showNotification('No data sources loaded', 'error');
            return;
        }
        
        let source = this.currentData.dataSources.find(s => s.id == sourceId);
        
        if (!source) {
            console.error('Source not found with ID:', sourceId);
            this.showNotification('Data source not found', 'error');
            return;
        }

        // Show confirmation modal
        document.getElementById('deleteDetails').innerHTML = `
            <div class="source-delete-item">
                <strong>${this.escapeHtml(source.name)}</strong>
                <br><small class="text-muted">${source.type} • ${source.file_size_formatted} • ${(source.estimated_rows !== null && source.estimated_rows !== undefined) ? source.estimated_rows.toLocaleString() : 0} rows</small>
            </div>
        `;

        const modalElement = document.getElementById('deleteModal');
        const modal = new bootstrap.Modal(modalElement);
        modal.show();

        // Store the source ID for confirmation
        const confirmBtn = document.getElementById('confirmDeleteBtn');
        confirmBtn.removeAttribute('data-source-ids');
        confirmBtn.setAttribute('data-source-id', source.id);
    }

    async confirmDelete() {
        const sourceId = document.getElementById('confirmDeleteBtn').getAttribute('data-source-id');
        const sourceIds = document.getElementById('confirmDeleteBtn').getAttribute('data-source-ids');
        
        // Handle bulk delete
        if (sourceIds) {
            const idsArray = sourceIds.split(',').map(id => id.trim());
            
            try {
                this.showLoading(true, 'Deleting data sources...', 'Please wait...');
                
                let successCount = 0;
                let errorCount = 0;
                const errors = [];
                
                for (const id of idsArray) {
                    try {
                        const response = await fetch(`/admin/api/data-ingestion/source/${id}`, {
                            method: 'DELETE',
                            headers: {
                                'Accept': 'application/json',
                                'Content-Type': 'application/json'
                            }
                        });
                        
                        const result = await response.json();
                        
                        if (result.success) {
                            successCount++;
                            const row = document.querySelector(`tr[data-source-id="${id}"]`);
                            if (row) row.remove();
                        } else {
                            errorCount++;
                            errors.push(`${id}: ${result.error}`);
                        }
                    } catch (error) {
                        errorCount++;
                        errors.push(`${id}: ${error.message}`);
                    }
                }
                
                this.selectedSources.clear();
                await this.loadDataIngestionStats();
                
                if (successCount > 0 && errorCount === 0) {
                    this.showNotification(`Successfully deleted ${successCount} data source(s)`, 'success');
                } else if (successCount > 0 && errorCount > 0) {
                    this.showNotification(`Deleted ${successCount} source(s), ${errorCount} failed: ${errors.join('; ')}`, 'warning');
                } else {
                    this.showNotification(`Failed to delete sources: ${errors.join('; ')}`, 'error');
                }
                
                const modalInstance = bootstrap.Modal.getInstance(document.getElementById('deleteModal'));
                if (modalInstance) modalInstance.hide();
                
            } catch (error) {
                this.showNotification('Failed to delete data sources: ' + error.message, 'error');
            } finally {
                this.showLoading(false);
            }
            return;
        }
        
        // Handle single delete
        if (!sourceId) {
            this.showNotification('No source selected for deletion', 'error');
            return;
        }

        try {
            this.showLoading(true, 'Deleting data source...', 'Please wait...');

            const response = await fetch(`/admin/api/data-ingestion/source/${sourceId}`, {
                method: 'DELETE',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();

            if (result.success) {
                this.showNotification(result.message, 'success');
                
                const row = document.querySelector(`tr[data-source-id="${sourceId}"]`);
                if (row) row.remove();
                
                await this.loadDataIngestionStats();
                
                const modalInstance = bootstrap.Modal.getInstance(document.getElementById('deleteModal'));
                if (modalInstance) modalInstance.hide();
            } else {
                this.showNotification('Error: ' + result.error, 'error');
            }
        } catch (error) {
            this.showNotification('Failed to delete data source: ' + error.message, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    async bulkDeleteSources() {
        if (this.selectedSources.size === 0) {
            this.showNotification('Please select data sources to delete', 'warning');
            return;
        }

        const selectedSourcesData = Array.from(this.selectedSources).map(sourceId => {
            return this.currentData.dataSources?.find(s => s.id == sourceId);
        }).filter(Boolean);

        document.getElementById('deleteDetails').innerHTML = `
            <div class="mb-2"><strong>Selected sources (${selectedSourcesData.length}):</strong></div>
            ${selectedSourcesData.map(source => `
                <div class="source-delete-item mb-2">
                    <strong>${this.escapeHtml(source.name)}</strong>
                    <br><small class="text-muted">${source.type} • ${source.file_size_formatted} • ${(source.estimated_rows !== null && source.estimated_rows !== undefined) ? source.estimated_rows.toLocaleString() : 0} rows</small>
                </div>
            `).join('')}
        `;

        const modal = new bootstrap.Modal(document.getElementById('deleteModal'));
        modal.show();

        const sourceIds = selectedSourcesData.map(s => s.id);
        const confirmBtn = document.getElementById('confirmDeleteBtn');
        confirmBtn.removeAttribute('data-source-id');
        confirmBtn.setAttribute('data-source-ids', sourceIds.join(','));
    }

    previewSource(sourceId) {
        window.open(`/ml-workflow/api/data-sources/${sourceId}/preview`, '_blank');
    }

    async refreshDataSources() {
        await this.loadDataSources();
        this.showNotification('Data sources refreshed', 'info');
    }

    // =============================================================================
    // User Management
    // =============================================================================

    async loadUserStats() {
        try {
            console.log('Loading user stats...');
            const response = await fetch('/admin/api/users/stats');
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();

            if (result.success) {
                this.updateUserStats(result.stats);
                this.renderDataSourcesByUser(result.stats.data_sources_by_user_detailed || {}, result.stats.data_sources_by_user || {});
                return result.stats;
            } else {
                this.showNotification('Error loading user statistics: ' + result.error, 'error');
                return null;
            }
        } catch (error) {
            console.error('Error loading user stats:', error);
            this.showNotification('Failed to load user statistics', 'error');
            return null;
        }
    }

    async getUserStatsData() {
        try {
            const response = await fetch('/admin/api/users/stats');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            const result = await response.json();
            return result.success ? result.stats : null;
        } catch (error) {
            console.error('Error getting user stats data:', error);
            return null;
        }
    }

    updateUserStats(stats) {
        // Update main User Management tab statistics
        document.getElementById('totalUsers').textContent = stats.total_users.toLocaleString();
        document.getElementById('activeUsers').textContent = stats.active_users.toLocaleString();
        document.getElementById('adminUsers').textContent = (stats.users_by_role?.admin || 0).toLocaleString();
        document.getElementById('onlineUsers').textContent = (stats.online_users || 0).toLocaleString();
        
        // Update Statistics & Analytics tab
        this.updateStatisticsAnalyticsTab(stats);
        
        this.createUserCharts(stats);
    }

    updateStatisticsAnalyticsTab(stats) {
        // Update Statistics & Analytics tab statistics
        const elements = {
            'totalUsersCount': stats.total_users || 0,
            'activeUsersCount': stats.active_users || 0,
            'adminUsersCount': stats.users_by_role?.admin || 0,
            'recentSignupsCount': stats.new_users_this_month || 0,
            'activity24h': stats.last_24h_logins || 0,
            'activityThisMonth': stats.new_users_this_month || 0,
            'activityDataSources': Object.keys(stats.data_sources_by_user || {}).length,
            'activityAverage': this.calculateAverageSourcesPerUser(stats)
        };

        for (const [elementId, value] of Object.entries(elements)) {
            const element = document.getElementById(elementId);
            if (element) {
                element.textContent = typeof value === 'number' ? value.toLocaleString() : value;
            }
        }
    }

    calculateAverageSourcesPerUser(stats) {
        const totalUsers = stats.total_users || 0;
        const dataSources = stats.data_sources_by_user || {};
        const totalSources = Object.values(dataSources).reduce((sum, count) => sum + count, 0);
        
        if (totalUsers === 0) return '0';
        
        const average = totalSources / totalUsers;
        return average.toFixed(1);
    }

    createUserCharts(stats) {
        this.createUserRoleChart(stats);
        this.createUserStatusChart(stats);
    }

    createUserRoleChart(stats) {
        const canvas = document.getElementById('userRoleChart');
        if (!canvas) {
            console.warn('User role chart canvas not found');
            return;
        }
        
        if (this.charts.userRoleChart) {
            this.charts.userRoleChart.destroy();
        }
        
        try {
            const ctx = canvas.getContext('2d');
            const roleData = stats.users_by_role || {admin: 0, user: 0};
            
            this.charts.userRoleChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Administrators', 'Regular Users'],
                    datasets: [{
                        data: [roleData.admin, roleData.user],
                        backgroundColor: ['#dc3545', '#0d6efd'],
                        borderWidth: 2,
                        borderColor: '#ffffff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                padding: 10,
                                usePointStyle: true
                            }
                        }
                    },
                    cutout: '60%'
                }
            });
        } catch (error) {
            console.error('Error creating user role chart:', error);
        }
    }

    createUserStatusChart(stats) {
        const canvas = document.getElementById('userStatusChart');
        if (!canvas) {
            console.warn('User status chart canvas not found');
            return;
        }
        
        if (this.charts.userStatusChart) {
            this.charts.userStatusChart.destroy();
        }
        
        try {
            const ctx = canvas.getContext('2d');
            const activeUsers = stats.active_users || 0;
            const inactiveUsers = (stats.total_users || 0) - activeUsers;
            
            this.charts.userStatusChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Active Users', 'Inactive Users'],
                    datasets: [{
                        data: [activeUsers, inactiveUsers],
                        backgroundColor: ['#198754', '#6c757d'],
                        borderWidth: 2,
                        borderColor: '#ffffff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                padding: 10,
                                usePointStyle: true
                            }
                        }
                    },
                    cutout: '60%'
                }
            });
        } catch (error) {
            console.error('Error creating user status chart:', error);
        }
    }

    renderDataSourcesByUser(dataSourcesByUserDetailed, dataSourcesByUserSimple) {
        const tbody = document.getElementById('dataSourcesByUserTableBody');
        
        if (!tbody) {
            console.error('Data sources by user table body not found');
            return;
        }
        
        const dataToUse = Object.keys(dataSourcesByUserDetailed || {}).length > 0 ? dataSourcesByUserDetailed : dataSourcesByUserSimple;
        
        if (!dataToUse || Object.keys(dataToUse).length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="4" class="text-center py-3 text-muted">
                        <i class="fas fa-database"></i> No data sources found
                    </td>
                </tr>
            `;
            return;
        }

        const users = Object.keys(dataToUse);
        
        const tableRows = users.map(user => {
            const userData = dataToUse[user];
            let count, size, lastActivity;
            
            if (typeof userData === 'object' && userData.count !== undefined) {
                count = userData.count;
                size = userData.size || 'N/A';
                lastActivity = userData.last_activity || 'Never';
            } else {
                count = userData;
                size = 'N/A';
                lastActivity = 'N/A';
            }
            
            return `
                <tr>
                    <td>
                        <div class="d-flex align-items-center">
                            <i class="fas fa-user text-muted me-2"></i>
                            <span class="fw-medium">${this.escapeHtml(user)}</span>
                        </div>
                    </td>
                    <td class="text-center">
                        <span class="badge bg-primary">${count}</span>
                    </td>
                    <td class="text-center">
                        <small class="text-muted">${size}</small>
                    </td>
                    <td class="text-center">
                        <small class="text-muted">${lastActivity}</small>
                    </td>
                </tr>
            `;
        });
        
        tbody.innerHTML = tableRows.join('');
    }

    async refreshUserStats() {
        await this.loadUserStats();
        this.showNotification('User statistics refreshed', 'info');
    }

    // =============================================================================
    // System Information Management
    // =============================================================================

    async loadSystemInfo() {
        try {
            // Build headers and include Authorization if a token is present in localStorage
            const headers = {
                'Accept': 'application/json'
            };

            const storedToken = localStorage.getItem('access_token');
            if (storedToken) {
                // If stored token already contains 'Bearer ', avoid double prefix
                headers['Authorization'] = storedToken.toLowerCase().startsWith('bearer ') ? storedToken : `Bearer ${storedToken}`;
            }

            const response = await fetch('/admin/api/system/info', {
                method: 'GET',
                // include credentials (cookies) in requests; 'include' is more permissive than 'same-origin'
                credentials: 'include',
                headers: headers
            });

            // If HTTP status is not OK, try to extract error information
            if (!response.ok) {
                let errMsg = `HTTP ${response.status}: ${response.statusText}`;
                try {
                    const errJson = await response.json();
                    errMsg = errJson.error || errJson.detail || JSON.stringify(errJson);
                } catch (e) {
                    try {
                        const text = await response.text();
                        if (text) errMsg = text.substring(0, 200);
                    } catch (e2) {
                        // ignore
                    }
                }
                this.showNotification('Error loading system info: ' + errMsg, 'error');
                return;
            }

            // Try to parse JSON and handle missing fields
            let result = null;
            try {
                result = await response.json();
            } catch (e) {
                this.showNotification('Error loading system info: invalid JSON response', 'error');
                return;
            }

            if (result && result.success) {
                // Some endpoints return `info` (we updated the API to do so)
                this.updateSystemInfo(result.info || result.system_info || {});
            } else {
                const msg = (result && (result.error || result.detail)) || 'Unknown error';
                this.showNotification('Error loading system info: ' + msg, 'error');
            }
        } catch (error) {
            this.showNotification('Failed to load system information', 'error');
        }
    }

    updateSystemInfo(info) {
        const memoryPercent = info.memory?.percent || 0;
        const cpuPercent = info.cpu?.percent || 0;
        const diskPercent = info.disk?.percent || 0;
        const uptime = info.uptime?.uptime_formatted || 'Unknown';

        document.getElementById('memoryUsage').textContent = `${memoryPercent.toFixed(1)}%`;
        document.getElementById('memoryDetails').textContent = `${info.memory?.used_formatted} / ${info.memory?.total_formatted}`;
        
        document.getElementById('cpuUsage').textContent = `${cpuPercent.toFixed(1)}%`;
        document.getElementById('cpuDetails').textContent = `${info.cpu?.count || 0} cores`;
        
        document.getElementById('diskUsage').textContent = `${diskPercent.toFixed(1)}%`;
        document.getElementById('diskDetails').textContent = `${info.disk?.used_formatted} / ${info.disk?.total_formatted}`;
        
        document.getElementById('systemUptime').textContent = uptime;
        document.getElementById('uptimeDetails').textContent = new Date(info.uptime?.boot_time).toLocaleDateString();

        document.getElementById('osInfo').textContent = `${info.system?.platform} ${info.system?.platform_version}`;
        document.getElementById('pythonVersion').textContent = info.system?.python_version || 'Unknown';
        document.getElementById('flaskVersion').textContent = info.system?.flask_version || 'Unknown';
        document.getElementById('serverStartTime').textContent = new Date(info.system?.server_start_time).toLocaleString();
        document.getElementById('processorCount').textContent = info.cpu?.count || 'Unknown';

        const storage = info.storage || {};
        document.getElementById('uploadsSize').textContent = storage.uploads?.size_formatted || '0 B';
        document.getElementById('cacheSize').textContent = storage.cache?.size_formatted || '0 B';
        document.getElementById('logsSize').textContent = storage.logs?.size_formatted || '0 B';
        document.getElementById('exportsSize').textContent = storage.exports?.size_formatted || '0 B';
        document.getElementById('totalFiles').textContent = storage.total_files || 0;

        document.getElementById('memoryProgressBar').style.width = `${memoryPercent}%`;
        document.getElementById('cpuProgressBar').style.width = `${cpuPercent}%`;
        document.getElementById('diskProgressBar').style.width = `${diskPercent}%`;

        this.updateProgressBarColor('memoryProgressBar', memoryPercent);
        this.updateProgressBarColor('cpuProgressBar', cpuPercent);
        this.updateProgressBarColor('diskProgressBar', diskPercent);
    }

    updateProgressBarColor(barId, percentage) {
        const bar = document.getElementById(barId);
        bar.className = bar.className.replace(/bg-\w+/, '');
        
        if (percentage > 80) {
            bar.classList.add('bg-danger');
        } else if (percentage > 60) {
            bar.classList.add('bg-warning');
        } else {
            bar.classList.add('bg-success');
        }
    }

    // =============================================================================
    // App Logs
    // =============================================================================
    async loadAppLogs() {
        try {
            const headers = { 'Accept': 'application/json' };
            const storedToken = localStorage.getItem('access_token');
            if (storedToken) headers['Authorization'] = storedToken.toLowerCase().startsWith('bearer ') ? storedToken : `Bearer ${storedToken}`;

            const resp = await fetch('/admin/api/logs/tail?lines=200', { credentials: 'include', headers });
            if (!resp.ok) {
                const j = await resp.json().catch(() => ({}));
                this.showNotification('Failed to load app logs: ' + (j.error || `HTTP ${resp.status}`), 'error');
                return;
            }
            const j = await resp.json();
            if (!j.success) {
                this.showNotification('Failed to load app logs: ' + (j.error || ''), 'error');
                return;
            }
            const el = document.getElementById('appLogContainer');
            if (el) el.textContent = j.tail || '';
        } catch (e) {
            console.error('Error loading app logs:', e);
            this.showNotification('Failed to load app logs', 'error');
        }
    }

    async downloadAppLog() {
        try {
            const headers = {};
            const storedToken = localStorage.getItem('access_token');
            if (storedToken) headers['Authorization'] = storedToken.toLowerCase().startsWith('bearer ') ? storedToken : `Bearer ${storedToken}`;

            const resp = await fetch('/admin/api/logs/download', { credentials: 'include', headers });
            if (!resp.ok) {
                const j = await resp.json().catch(() => ({}));
                this.showNotification('Failed to download app log: ' + (j.error || `HTTP ${resp.status}`), 'error');
                return;
            }
            const blob = await resp.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'fastapi_app.log';
            a.click();
            URL.revokeObjectURL(url);
        } catch (e) {
            console.error('Error downloading app log:', e);
            this.showNotification('Failed to download app log', 'error');
        }
    }

    // =============================================================================
    // System Maintenance
    // =============================================================================
    async loadMaintenanceStatus() {
        try {
            const headers = { 'Accept': 'application/json' };
            const storedToken = localStorage.getItem('access_token');
            if (storedToken) headers['Authorization'] = storedToken.toLowerCase().startsWith('bearer ') ? storedToken : `Bearer ${storedToken}`;

            const resp = await fetch('/admin/api/maintenance/status', { credentials: 'include', headers });
            if (!resp.ok) {
                const j = await resp.json().catch(() => ({}));
                this.showNotification('Failed to load maintenance status: ' + (j.error || `HTTP ${resp.status}`), 'error');
                this.showMaintenanceError('Failed to load maintenance status');
                return;
            }
            const j = await resp.json();
            if (!j.success) {
                this.showNotification('Failed to load maintenance status: ' + (j.error || ''), 'error');
                this.showMaintenanceError('Failed to load maintenance status');
                return;
            }
            
            this.updateMaintenanceStatus(j.status);
        } catch (e) {
            console.error('Error loading maintenance status:', e);
            this.showNotification('Failed to load maintenance status', 'error');
            this.showMaintenanceError('Network error loading maintenance status');
        }
    }

    async runMaintenance() {
        const runButton = document.getElementById('runMaintenanceBtn');
        const originalText = runButton.innerHTML;
        
        try {
            // Show loading state
            runButton.disabled = true;
            runButton.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Running...';
            
            this.showMaintenanceAlert('Starting system maintenance...', 'info');
            
            const headers = { 
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            };
            const storedToken = localStorage.getItem('access_token');
            if (storedToken) headers['Authorization'] = storedToken.toLowerCase().startsWith('bearer ') ? storedToken : `Bearer ${storedToken}`;

            const resp = await fetch('/admin/api/maintenance/run', { 
                method: 'POST',
                credentials: 'include', 
                headers 
            });
            
            if (!resp.ok) {
                const j = await resp.json().catch(() => ({}));
                throw new Error(j.error || j.detail || `HTTP ${resp.status}`);
            }
            
            const j = await resp.json();
            if (!j.success) {
                throw new Error(j.error || 'Maintenance failed');
            }
            
            this.showMaintenanceAlert('Maintenance completed successfully!', 'success');
            this.displayMaintenanceResults(j.results);
            this.showNotification('System maintenance completed successfully', 'success');
            
            // Reload status to get updated stats
            setTimeout(() => this.loadMaintenanceStatus(), 1000);
            
        } catch (e) {
            console.error('Error running maintenance:', e);
            this.showMaintenanceAlert(`Maintenance failed: ${e.message}`, 'danger');
            this.showNotification('Failed to run maintenance: ' + e.message, 'error');
        } finally {
            // Reset button state
            runButton.disabled = false;
            runButton.innerHTML = originalText;
        }
    }

    updateMaintenanceStatus(status) {
        // Update status cards
        const systemStatusEl = document.getElementById('maintenanceSystemStatus');
        const totalRunsEl = document.getElementById('maintenanceTotalRuns');
        const totalFilesEl = document.getElementById('maintenanceTotalFiles');
        const lastRunEl = document.getElementById('maintenanceLastRun');

        if (systemStatusEl) {
            const isEnabled = status.configuration?.system_cleanup_enabled;
            systemStatusEl.innerHTML = isEnabled 
                ? '<span class="text-success"><i class="fas fa-check-circle me-1"></i>Enabled</span>'
                : '<span class="text-warning"><i class="fas fa-exclamation-triangle me-1"></i>Disabled</span>';
        }

        if (totalRunsEl) {
            totalRunsEl.textContent = status.total_runs || 0;
        }

        if (totalFilesEl) {
            totalFilesEl.textContent = status.total_files_removed || 0;
        }

        if (lastRunEl) {
            const lastRun = status.last_run;
            if (lastRun) {
                const date = new Date(lastRun);
                lastRunEl.innerHTML = `<small>${date.toLocaleDateString()}<br>${date.toLocaleTimeString()}</small>`;
            } else {
                lastRunEl.innerHTML = '<span class="text-muted">Never</span>';
            }
        }

        // Update configuration display
        this.updateMaintenanceConfig(status.configuration);
    }

    updateMaintenanceConfig(config) {
        const configContainer = document.getElementById('maintenanceConfig');
        if (!configContainer || !config) return;

        const operations = config.operations_enabled || {};
        const operationDetails = config.operation_details || {};
        
        const configHtml = `
            <div class="col-md-6">
                <h6>System Settings</h6>
                <div class="mb-2">
                    <span class="badge ${config.system_cleanup_enabled ? 'bg-success' : 'bg-warning'} me-2">
                        <i class="fas ${config.system_cleanup_enabled ? 'fa-check' : 'fa-times'} me-1"></i>
                        Cleanup ${config.system_cleanup_enabled ? 'Enabled' : 'Disabled'}
                    </span>
                </div>
                <div class="mb-2">
                    <span class="badge ${config.cleanup_on_startup ? 'bg-info' : 'bg-secondary'} me-2">
                        <i class="fas fa-power-off me-1"></i>
                        Startup Cleanup ${config.cleanup_on_startup ? 'On' : 'Off'}
                    </span>
                </div>
                <div class="mb-2">
                    <span class="badge ${config.scheduler_enabled ? 'bg-primary' : 'bg-secondary'} me-2">
                        <i class="fas fa-clock me-1"></i>
                        Scheduler ${config.scheduler_enabled ? 'Active' : 'Inactive'}
                    </span>
                </div>
            </div>
            <div class="col-md-6">
                <h6>Operation Types <small class="text-muted">(Controlled by Config)</small></h6>
                ${Object.entries(operations).map(([op, enabled]) => {
                    const details = operationDetails[op] || {};
                    const statusClass = enabled ? 'text-success' : 'text-muted';
                    return `
                        <div class="mb-2">
                            <div class="form-check form-check-inline">
                                <input class="form-check-input" type="checkbox" ${enabled ? 'checked' : ''} disabled>
                                <label class="form-check-label text-capitalize ${statusClass}">
                                    <strong>${op}</strong>
                                </label>
                            </div>
                            ${enabled && details.max_files !== undefined ? `
                                <div class="ms-4">
                                    <small class="text-muted">
                                        Keep ${details.max_files} files, ${details.max_age_days} days max age
                                    </small>
                                </div>
                            ` : ''}
                            ${!enabled ? `
                                <div class="ms-4">
                                    <small class="text-muted">Disabled in configuration</small>
                                </div>
                            ` : ''}
                        </div>
                    `;
                }).join('')}
            </div>
        `;
        
        configContainer.innerHTML = configHtml;
    }

    displayMaintenanceResults(results) {
        const resultsContainer = document.getElementById('maintenanceResults');
        if (!resultsContainer || !results) return;

        const resultsHtml = Object.entries(results).map(([operation, result]) => {
            const statusClass = result.success ? 'success' : 'danger';
            const statusIcon = result.success ? 'fa-check-circle' : 'fa-times-circle';
            
            // Get operation details from result
            let detailsHtml = '';
            if (result.details && result.details.criteria) {
                const criteria = result.details.criteria;
                const filesInDir = result.details.remaining_files || 0;
                const totalFiles = filesInDir + result.files_removed;
                
                detailsHtml = `
                    <div class="col-12 mt-2">
                        <small class="text-muted">
                            <strong>Config:</strong> Keep ${criteria.max_files} newest files, remove files older than ${criteria.max_age_days} days<br>
                            <strong>Found:</strong> ${totalFiles} files total, ${result.files_removed} removed, ${filesInDir} remaining
                            ${criteria.remove_data_sources ? ' (with database cleanup)' : ''}
                        </small>
                    </div>
                `;
            }
            
            return `
                <div class="row mb-3 p-3 border rounded">
                    <div class="col-md-4">
                        <span class="text-${statusClass}">
                            <i class="fas ${statusIcon} me-2"></i><strong>${operation}</strong>
                        </span>
                    </div>
                    <div class="col-md-3 text-center">
                        <span class="badge bg-${statusClass}">${result.files_removed} files removed</span>
                        ${result.details && result.details.data_sources_removed ? 
                            `<br><span class="badge bg-info mt-1">${result.details.data_sources_removed} DB records</span>` : ''
                        }
                    </div>
                    <div class="col-md-5">
                        ${result.success 
                            ? '<small class="text-success">✓ Completed successfully</small>' 
                            : `<small class="text-danger">✗ ${result.error_message || 'Operation failed'}</small>`
                        }
                        ${result.error_message === 'Disabled in configuration' ? 
                            '<br><small class="text-muted">(Controlled by config)</small>' : ''
                        }
                    </div>
                    ${detailsHtml}
                </div>
            `;
        }).join('');

        resultsContainer.innerHTML = `
            <div class="alert alert-light">
                <h6 class="mb-3"><i class="fas fa-list-check me-2"></i>Latest Operation Results</h6>
                ${resultsHtml}
                <hr class="mt-4">
                <small class="text-muted">
                    <i class="fas fa-info-circle me-1"></i>
                    <strong>Operation Types:</strong> All cleanup operations are controlled by configuration settings. 
                    Files are removed when they exceed the maximum count or age limits defined in the system configuration.
                </small>
            </div>
        `;
    }

    showMaintenanceAlert(message, type) {
        const alertEl = document.getElementById('maintenanceAlert');
        const alertTextEl = document.getElementById('maintenanceAlertText');
        
        if (alertEl && alertTextEl) {
            alertEl.className = `alert alert-${type}`;
            alertEl.style.display = 'block';
            alertTextEl.textContent = message;
            
            // Auto-hide info messages after 5 seconds
            if (type === 'info') {
                setTimeout(() => {
                    alertEl.style.display = 'none';
                }, 5000);
            }
        }
    }

    showMaintenanceError(message) {
        // Show error state in status cards
        ['maintenanceSystemStatus', 'maintenanceTotalRuns', 'maintenanceTotalFiles', 'maintenanceLastRun'].forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.innerHTML = '<span class="text-danger"><i class="fas fa-exclamation-triangle"></i></span>';
            }
        });

        // Show error in config section
        const configContainer = document.getElementById('maintenanceConfig');
        if (configContainer) {
            configContainer.innerHTML = `
                <div class="col-12">
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        ${message}
                    </div>
                </div>
            `;
        }
    }

    // =============================================================================
    // Utility Functions
    // =============================================================================

    showLoading(show, title = 'Loading...', details = 'Please wait...') {
        const overlay = document.getElementById('loadingOverlay');
        
        if (show) {
            document.getElementById('loadingMessage').textContent = title;
            document.getElementById('loadingDetails').textContent = details;
            overlay.style.display = 'flex';
        } else {
            overlay.style.display = 'none';
        }
    }

    async refreshDashboard() {
        this.showLoading(true, 'Refreshing Dashboard...', 'Loading the latest data...');
        try {
            await this.loadInitialData();
            this.showNotification('Dashboard refreshed successfully', 'success');
        } catch (error) {
            this.showNotification('Failed to refresh dashboard', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    // Open the Quality Help modal (called from inline onclick in the template)
    showQualityHelp() {
        try {
            const modalEl = document.getElementById('qualityHelpModal');
            if (!modalEl) {
                this.showNotification('Quality help is not available in this view', 'warning');
                return;
            }

            const modal = new bootstrap.Modal(modalEl);
            modal.show();
        } catch (e) {
            console.error('Error showing quality help modal:', e);
            this.showNotification('Unable to open quality help', 'error');
        }
    }
}

// Global functions for HTML onclick handlers
async function toggleUserStatus(userId, username, currentStatus) {
    if (window.adminDashboard) {
        await window.adminDashboard.toggleUserStatus(userId, username, currentStatus);
    }
}

async function editUser(userId) {
    if (window.adminDashboard) {
        window.adminDashboard.editUser(userId);
    }
}

async function deleteUser(userId, username) {
    if (window.adminDashboard) {
        await window.adminDashboard.deleteUser(userId, username);
    }
}

// Global utility functions for HTML
function togglePasswordVisibility(inputId) {
    const input = document.getElementById(inputId);
    const icon = document.getElementById(inputId + 'Icon');
    
    if (input.type === 'password') {
        input.type = 'text';
        icon.className = 'fas fa-eye-slash';
    } else {
        input.type = 'password';
        icon.className = 'fas fa-eye';
    }
}

function resetCreateUserForm() {
    if (window.adminDashboard) {
        window.adminDashboard.resetCreateUserForm();
    }
}

function refreshDashboard() {
    if (window.adminDashboard) {
        // Reload current tab content
        const activeTab = document.querySelector('.admin-tabs .nav-link.active');
        if (activeTab) {
            const target = activeTab.getAttribute('data-bs-target');
            window.adminDashboard.onTabChange(target);
        }
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.adminDashboard = new AdminDashboard();
});

// Global maintenance function for HTML onclick
function runMaintenance() {
    if (window.adminDashboard) {
        window.adminDashboard.runMaintenance();
    }
}